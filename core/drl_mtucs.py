"""
DRL-MTUCS 主模块（含完整测试版本）
-------------------------------------
整合：
    - 高层任务分配器 High-Level Allocator
    - 低层 UAV 策略网络 Low-Level Policy
    - 时间预测器 Temporal Predictor
    - PPO Trainer （可使用 mock）
    - UAV 环境（可使用 mock）

说明：
- 该文件为 project-level glue：把 high/low/temporal 三个模块与环境和 PPO 连接起来。
- Dummy 测试会用到 fabricated map_obs 和 goal（用于兼容 LowLevelUAVPolicy.act 接口）。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# 将项目根目录加入 Python 路径（以便从 core/ 下直接运行时能 import models/ 等）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ============================
# 主 DRL_MTUCS 类
# ============================
class DRL_MTUCS:
    def __init__(self, env, high_level, low_level, temporal_predictor, config):
        """
        env: environment instance (must implement reset(), step(assign, actions), is_emergency(...))
        high_level: MultiHeadHighLevelAllocator instance (callable: high_level(uav_feats, task_feats) -> logits (H,U,T))
        low_level: LowLevelUAVPolicy instance (must implement act(vector_obs, map_obs, goal) ...)
        temporal_predictor: TemporalPredictor instance (expects sequence input: [B, seq_len, input_dim])
        config: dict with training hyperparams, must contain config["train"]["gamma"]
        """
        self.env = env
        self.high_level = high_level
        self.low_level = low_level
        self.temporal_predictor = temporal_predictor

        self.cfg = config
        self.gamma = config.get("train", {}).get("gamma", 0.99)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move networks to device
        self.high_level.to(self.device)
        self.low_level.to(self.device)
        self.temporal_predictor.to(self.device)

        # default map size used for dummy testing (if real env provides maps, use those)
        self._dummy_map_channels = 1
        self._dummy_map_size = 20

    def run_episode(self, ppo):
        """
        Run one episode in the environment, collecting transitions into PPO.
        Returns (episode_reward, metrics_dict)
        """
        obs = self.env.reset()
        done = False

        ep_reward = 0.0
        step = 0

        emergency_served = 0
        total_emergency = 0
        last_intrinsic = 0.0

        while not done:
            # ------------------ Prepare features ------------------
            # Expect obs dict from env.reset()/env.step to contain:
            #  obs["uav"] -> np.array (U, uav_dim)
            #  obs["task"] -> np.array (T, task_dim)
            uav_np = np.asarray(obs["uav"])
            task_np = np.asarray(obs["task"])

            uav_features = torch.tensor(uav_np, dtype=torch.float32, device=self.device)    # (U, uav_dim)
            task_features = torch.tensor(task_np, dtype=torch.float32, device=self.device)  # (T, task_dim)

            # ------------------ High-level allocation ------------------
            with torch.no_grad():
                logits = self.high_level(uav_features, task_features)  # (H, U, T)
                # ensure at least 2 heads
                if logits.shape[0] < 2:
                    # fallback: replicate the first head
                    logits = logits.repeat(2, 1, 1)
                head0 = F.softmax(logits[0], dim=-1)  # normal tasks
                head1 = F.softmax(logits[1], dim=-1)  # emergency tasks

                assign_normal = head0.argmax(dim=-1).cpu().numpy()     # (U,)
                assign_emergency = head1.argmax(dim=-1).cpu().numpy()  # (U,)

            # fuse decisions: prefer emergency assignment if valid
            assignment = []
            for i in range(len(assign_normal)):
                chosen = assign_emergency[i] if self.env.is_emergency(assign_emergency[i]) else assign_normal[i]
                assignment.append(int(chosen))
            assignment = np.array(assignment, dtype=np.int32)  # (U,)

            # ------------------ Low-level actions ------------------
            # LowLevelUAVPolicy interface: act(vector_obs, map_obs, goal, deterministic=False)
            # We need to provide per-UAV vector_obs, map_obs and goal.
            # For dummy/test env we fabricate map_obs and goal; real env should supply them.

            # vector_obs: use uav_features (U, uav_dim)
            vector_obs = uav_features  # Tensor on device

            # fabricate simple map_obs: zeros (U, C, H, W)
            map_obs = torch.zeros(
                (vector_obs.shape[0], self._dummy_map_channels, self._dummy_map_size, self._dummy_map_size),
                dtype=torch.float32,
                device=self.device
            )

            # fabricate goal vectors: here we use zeros with same dim as vector_obs last dim (or allow env to provide)
            goal_dim = vector_obs.shape[-1]
            goal = torch.zeros((vector_obs.shape[0], goal_dim), dtype=torch.float32, device=self.device)

            # call policy.act to get continuous actions and log_probs
            # action: Tensor (U, action_dim)
            with torch.no_grad():
                action_tensor, log_prob = self.low_level.act(vector_obs, map_obs, goal, deterministic=False)
            # convert actions to numpy for env
            actions = action_tensor.cpu().numpy()  # shape (U, action_dim) - environment should accept this format

            # ------------------ Temporal predictor (intrinsic reward) ------------------
            # The temporal predictor expects a sequence: [B, seq_len, input_dim]
            # For dummy testing, we create a seq_len=1 history from current vector_obs
            seq = vector_obs.unsqueeze(1)  # (U, 1, input_dim)
            with torch.no_grad():
                pred = self.temporal_predictor(seq)  # (U, output_dim)
            # compute intrinsic reward as negative error between predicted next-state and current state
            # if dims mismatch, reduce accordingly
            try:
                intrinsic_tensor = -torch.norm(pred - vector_obs, dim=-1)  # (U,)
                intrinsic_reward = float(intrinsic_tensor.mean().cpu().item())
            except Exception:
                # fallback: scalar intrinsic
                intrinsic_reward = 0.0
            last_intrinsic = intrinsic_reward

            # ------------------ Step environment ------------------
            # env.step expects (assignment, actions). We pass numpy arrays:
            next_obs, reward, done, info = self.env.step(assignment, actions)

            ep_reward += float(reward)
            step += 1

            emergency_served += info.get("emergency_served", 0)
            total_emergency += info.get("total_emergency", 0)

            # ------------------ Store transition into PPO ------------------
            # Convert tensors to CPU numpy where appropriate or keep tensors per your PPO impl.
            # Here we pass tensors for convenience.
            ppo.store_transition(
                uav_features=uav_features.detach().cpu().numpy(),
                task_features=task_features.detach().cpu().numpy(),
                action_uav_task=assignment,
                action_low=action_tensor.detach().cpu().numpy(),
                reward=float(reward) + intrinsic_reward,
                done=done
            )

            if ppo.is_update_time():
                ppo.update()

            obs = next_obs

        # Episode finished
        metrics = {
            "reward": ep_reward,
            "steps": step,
            "intrinsic": last_intrinsic,
            "emergency_rate": emergency_served / (total_emergency + 1e-8),
            "avg_aoi": info.get("avg_aoi", 0),
        }

        return ep_reward, metrics


# ============================
# 下面部分：Dummy Env + Mock PPO + 测试入口
# ============================
class UAVEnvironmentMock:
    """
    Minimal mock environment used for testing DRL_MTUCS integration.
    It returns observations in the dict format expected by DRL_MTUCS.
    """

    def __init__(self, num_uav=3, num_task=6, uav_dim=8, task_dim=10):
        self.num_uav = num_uav
        self.num_task = num_task
        self.uav_obs_dim = uav_dim
        self.task_obs_dim = task_dim
        self.action_dim = 2  # low-level continuous action dim (e.g., vx, vy)

        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        self.step_count = 0
        return {
            "uav": np.random.randn(self.num_uav, self.uav_obs_dim).astype(np.float32),
            "task": np.random.randn(self.num_task, self.task_obs_dim).astype(np.float32)
        }

    def step(self, assignment, actions):
        """
        assignment: (U,) numpy ints, chosen task indices
        actions: (U, action_dim) numpy floats
        """
        self.step_count += 1

        obs = {
            "uav": np.random.randn(self.num_uav, self.uav_obs_dim).astype(np.float32),
            "task": np.random.randn(self.num_task, self.task_obs_dim).astype(np.float32)
        }
        # simple reward: positive for taking any action, small noise
        reward = float(np.random.randn() * 0.1 + 1.0)
        done = self.step_count >= self.max_steps

        info = {
            "emergency_served": int(np.random.randint(0, 2)),
            "total_emergency": 1,
            "avg_aoi": float(np.random.rand())
        }
        return obs, reward, done, info

    def is_emergency(self, task_id):
        # simple rule for mock: any task_id divisible by 3 is emergency
        try:
            return int(task_id) % 3 == 0
        except Exception:
            return False


class PPOTrainerMock:
    """
    Minimal PPO stub that collects transitions (no learning).
    """

    def __init__(self):
        self.buffer = []

    def store_transition(self, **kwargs):
        self.buffer.append(kwargs)

    def is_update_time(self):
        return False

    def update(self):
        # no-op for mock
        self.buffer = []


# ============================
# Dummy test
# ============================
def _dummy_test():
    print(">>> Running DRL_MTUCS dummy test ... (this will create and run mock components)")

    # import concrete module classes from models (assumes project structure)
    from models.high_level_allocator import MultiHeadHighLevelAllocator
    from models.low_level_uav_policy import LowLevelUAVPolicy
    from models.temporal_predictor import TemporalPredictor

    # create mock environment
    env = UAVEnvironmentMock(num_uav=3, num_task=8, uav_dim=12, task_dim=10)

    # create networks
    high = MultiHeadHighLevelAllocator(
        uav_dim=env.uav_obs_dim,
        task_dim=env.task_obs_dim,
        embed_dim=64,
        num_heads=2
    )

    low = LowLevelHighLevel = None
    # LowLevelUAVPolicy constructor signature: (vector_obs_dim, goal_dim, action_dim, ...)
    low = LowLevelUAVPolicy(
        vector_obs_dim=env.uav_obs_dim,
        goal_dim=env.uav_obs_dim,
        action_dim=env.action_dim,
        map_size=20,
        map_channels=1,
        hidden_dim=128
    )

    # temporal predictor expects sequences: choose seq input_dim = uav_obs_dim
    temp = TemporalPredictor(input_dim=env.uav_obs_dim, output_dim=env.uav_obs_dim, hidden_size=128, num_layers=2)

    config = {"train": {"gamma": 0.99}}

    agent = DRL_MTUCS(env, high, low, temp, config)
    ppo = PPOTrainerMock()

    reward, metrics = agent.run_episode(ppo)

    print("Episode Reward:", reward)
    print("Metrics:", metrics)
    print(">>> DRL_MTUCS dummy test passed!")


if __name__ == "__main__":
    _dummy_test()
