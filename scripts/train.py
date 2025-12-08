import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
from collections import deque

import os
import sys

# --- 设置项目根目录，确保模块可导入 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- 正确导入你自己写的模块 ---
from models.high_level_allocator import MultiHeadHighLevelAllocator
from core.intrinsic_reward import IntrinsicRewardModule   # ← 修复了 instrinsic → intrinsic
# low_level_uav_policy 你还没写，用 Dummy 代替
# from models.low_level_uav_policy import LowLevelUAVPolicy


# ============================================================
# Placeholder Dummy 模块（你之后替换成真实环境和低层策略）
# ============================================================

# class DummyLowLevelPolicy(nn.Module):
#     def __init__(self, obs_dim, action_dim):
#         super().__init__()
#         self.actor = nn.Linear(obs_dim, action_dim)
#         self.critic = nn.Linear(obs_dim, 1)

#     def forward(self, obs):
#         return self.actor(obs)

#     def get_value(self, obs):
#         return self.critic(obs)


# class DummyDRLMTUCSEnvironment:
#     def __init__(self, num_uavs=5, high_action_dim=10):
#         self.num_uavs = num_uavs
#         self.high_action_dim = high_action_dim
#         self.global_obs_dim = 256
#         self.low_obs_dim = 128
#         self.low_action_dim = 4

#     def reset(self):
#         global_obs = torch.randn(self.global_obs_dim)
#         low_obs = [torch.randn(self.low_obs_dim) for _ in range(self.num_uavs)]
#         return global_obs, low_obs

#     def step(self, low_actions):
#         next_global_obs = torch.randn(self.global_obs_dim)
#         next_low_obs = [torch.randn(self.low_obs_dim) for _ in range(self.num_uavs)]
#         reward = torch.rand(1).item() * 10 - 5
#         done = False
#         return next_global_obs, next_low_obs, reward, done

#     def apply_high_level_allocation(self, allocation_index):
#         pass


# ============================================================
# 分层 PPO 代理
# ============================================================
class HierarchicalPPOAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 高层分配器（用 env 信息尽量合理初始化） ---
        # 如果 global_obs 用不同组织方式，请后续替换真实维度
        inferred_uav_dim = max(1, env.global_obs_dim // max(1, env.num_uavs))
        inferred_task_dim = inferred_uav_dim  # 占位：可根据真实 task_dim 修改

        self.high_allocator = MultiHeadHighLevelAllocator(
            uav_dim=inferred_uav_dim,
            task_dim=inferred_task_dim,
            embed_dim=64,
            num_heads=2,
            hidden_dim=128
        ).to(self.device)

        # --- 内在奖励模块 ---
        self.intrinsic_reward_module = IntrinsicRewardModule(
            obs_dim=env.low_obs_dim,
            action_dim=env.low_action_dim,
            feature_dim=args.feature_dim
        ).to(self.device)

        # --- 低层策略（暂用 dummy） ---
        self.low_policy = DummyLowLevelPolicy(env.low_obs_dim, env.low_action_dim).to(self.device)

        # --- 优化器 ---
        self.optimizer_high = optim.Adam(self.high_allocator.parameters(), lr=args.lr_high, eps=args.eps)
        self.optimizer_low = optim.Adam(self.low_policy.parameters(), lr=args.lr_low, eps=args.eps)
        self.optimizer_irm = optim.Adam(self.intrinsic_reward_module.parameters(), lr=args.lr_irm, eps=args.eps)

        # --- PPO 超参数 ---
        self.gamma = args.gamma
        self.tau = args.tau
        self.clip_param = args.clip_param
        self.ppo_epochs = args.ppo_epochs
        self.entropy_coef = args.entropy_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm

    def collect_trajectory(self):
        """
        与环境交互，收集轨迹 for PPO + IRM。
        - high_buffer: 存高层周期性决策信息（obs numpy, action tensor, log_prob tensor, value tensor, reward float, done）
        - low_buffer: 存 (obs_tensor, action_tensor, next_obs_tensor) 用于 IRM 重新计算 loss（保留 tensor -> detach when storing）
        返回 (obs_batch, actions, old_log_probs, values, returns, advantages, low_buffer)
        """
        high_buffer = []
        low_buffer = []

        global_obs, low_obs_list = self.env.reset()
        global_obs = global_obs.to(self.device)

        step_count = 0
        current_high_step = -1
        done = False

        while step_count < self.args.trajectory_len:
            # 1) 高层决策（周期性）
            if step_count % self.args.high_level_period == 0:
                high_action, high_log_prob = self.high_allocator.act(global_obs.unsqueeze(0))
                high_value = self.high_allocator.get_value(global_obs.unsqueeze(0))

                # 标准化 high_action：允许 tensor 或 list
                if isinstance(high_action, (list, tuple)):
                    high_action_tensor = torch.tensor(high_action).to(self.device)
                else:
                    high_action_tensor = high_action.detach().to(self.device)

                high_log_prob = high_log_prob.detach().to(self.device)
                high_value = high_value.detach().to(self.device)

                high_buffer.append({
                    "obs": global_obs.detach().cpu().numpy(),
                    "action": high_action_tensor.detach().cpu().numpy(),   # 存 numpy（flat）
                    "log_prob": high_log_prob.detach(),                    # 存 tensor（后续 stack）
                    "value": high_value.detach(),                          # 存 tensor
                    "reward": 0.0,
                    "done": False
                })
                current_high_step = len(high_buffer) - 1

                # 将高层分配应用到 env（dummy 环境应该接受 list/array）
                try:
                    # 如果 high_action_tensor 是向量（per-UAV），将其转为 list
                    self.env.apply_high_level_allocation(high_action_tensor.detach().cpu().tolist())
                except Exception:
                    # 兼容性：如果 env 只接受 scalar，尝试第一个元素
                    try:
                        self.env.apply_high_level_allocation(int(high_action_tensor.detach().cpu().flatten()[0]))
                    except Exception:
                        pass

            # 2) 低层执行：为每架 UAV 取动作（这里用 DummyLowLevelPolicy）
            low_actions = []
            for obs in low_obs_list:
                act = self.low_policy(obs.to(self.device))
                low_actions.append(act.detach().cpu().numpy())

            # 3) 环境 step
            next_global_obs, next_low_obs_list, extrinsic_reward, done = self.env.step(low_actions)

            # 4) 计算 IRM（只用第0个 UAV 的数据作为示例）
            obs_tensor = low_obs_list[0].unsqueeze(0).float().to(self.device)
            act_tensor = torch.tensor(low_actions[0]).unsqueeze(0).float().to(self.device)
            next_obs_tensor = next_low_obs_list[0].unsqueeze(0).float().to(self.device)

            # 这里 compute_intrinsic_reward_and_loss 返回 (intrinsic_reward_tensor, forward_loss_tensor)
            r_i, irm_loss_fwd = self.intrinsic_reward_module.compute_intrinsic_reward_and_loss(
                obs_tensor, act_tensor, next_obs_tensor,
                eta=self.args.eta_irm, forward_loss_weight=self.args.loss_irm_weight
            )

            intrinsic_reward = r_i.mean().item()
            total_reward = extrinsic_reward + intrinsic_reward

            # 5) 存 low buffer（存为 cpu tensor，后面重新 forward）
            low_buffer.append({
                "obs": obs_tensor.detach().cpu(),
                "action": act_tensor.detach().cpu(),
                "next_obs": next_obs_tensor.detach().cpu()
            })

            # 6) 更新高层 reward 累积
            if current_high_step != -1:
                high_buffer[current_high_step]["reward"] += total_reward
                if done:
                    high_buffer[current_high_step]["done"] = True

            # 7) 状态推进
            global_obs = next_global_obs.to(self.device)
            low_obs_list = next_low_obs_list
            step_count += 1

            if done:
                break

        # --- bootstrap value ---
        with torch.no_grad():
            final_value = self.high_allocator.get_value(global_obs.unsqueeze(0)).detach().cpu().numpy() if not done else np.array([[0.0]])

        # --- convert high buffer into batched tensors (use numpy.stack for obs to avoid warning) ---
        if len(high_buffer) == 0:
            # 避免空 buffer 出错，返回 empty placeholders
            obs_batch = torch.zeros(0).to(self.device)
            actions = torch.zeros(0).long().to(self.device)
            old_log_probs = torch.zeros(0).to(self.device)
            values = torch.zeros(0).to(self.device)
            returns = torch.zeros(0).to(self.device)
            advantages = torch.zeros(0).to(self.device)
            return obs_batch, actions, old_log_probs, values, returns, advantages, low_buffer

        obs_np = np.stack([np.asarray(d["obs"]) for d in high_buffer], axis=0)
        obs_batch = torch.from_numpy(obs_np).float().to(self.device)

        # actions stored as numpy arrays (possibly vector per high action)
        actions_np = np.stack([np.asarray(d["action"]) for d in high_buffer], axis=0)
        actions = torch.from_numpy(actions_np).to(self.device)   # keep numeric type; later cast in update if needed

        # log_probs & values are tensors kept in buffer; stack them
        old_log_probs = torch.stack([d["log_prob"].detach() for d in high_buffer], dim=0).to(self.device)
        values = torch.stack([d["value"].detach() for d in high_buffer], dim=0).to(self.device)

        rewards = torch.tensor([d["reward"] for d in high_buffer], dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor([d["done"] for d in high_buffer], dtype=torch.float32).unsqueeze(1).to(self.device)

        returns, advantages = self._compute_gae_and_returns(rewards, values, dones, final_value)

        return obs_batch, actions, old_log_probs, values, returns, advantages, low_buffer

    def _compute_gae_and_returns(self, rewards, values, dones, final_value):
        """
        Compute GAE and returns.
        - rewards: [T,1]
        - values:  [T,1]
        - dones:   [T,1]
        - final_value: numpy array shape (1,1) or tensor
        """
        T = rewards.size(0)
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)

        # convert final_value to tensor on device
        next_value = torch.tensor(final_value).float().to(self.device)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].item()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1].item()
                next_val = values[t+1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.tau * next_non_terminal * gae
            returns[t] = gae + values[t]
            advantages[t] = gae

        return returns, advantages

    def update_high(self, obs, actions, old_log_probs, values, returns, advantages):
        """
        PPO update for high-level allocator.
        - actions may be multi-dimensional (per-high-step vector). We forward it to evaluate_actions of allocator.
        """
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = 0.0
        value_loss = 0.0
        for _ in range(self.ppo_epochs):
            new_values, new_log_probs, entropy = self.high_allocator.evaluate_actions(obs, actions)
            # ensure shapes: new_log_probs and old_log_probs same shape
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer_high.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.high_allocator.parameters(), self.max_grad_norm)
            self.optimizer_high.step()

        return policy_loss.item(), value_loss.item()

    def update_low_and_irm(self, low_buffer):
        """
        Recompute IRM forward losses from stored low_buffer (each entry has cpu tensors for obs/action/next_obs).
        This ensures the IRM loss retains a proper computation graph for backprop.
        """
        if len(low_buffer) == 0:
            return 0.0, 0.0, 0.0

        irm_losses = []
        for d in low_buffer:
            obs = d["obs"].to(self.device).float()
            action = d["action"].to(self.device).float()
            next_obs = d["next_obs"].to(self.device).float()

            _, irm_loss_fwd = self.intrinsic_reward_module.compute_intrinsic_reward_and_loss(
                obs, action, next_obs,
                eta=self.args.eta_irm, forward_loss_weight=self.args.loss_irm_weight
            )
            irm_losses.append(irm_loss_fwd)

        irm_loss = torch.stack(irm_losses).mean()
        self.optimizer_irm.zero_grad()
        irm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.intrinsic_reward_module.parameters(), self.max_grad_norm)
        self.optimizer_irm.step()

        return irm_loss.item(), 0.0, 0.0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=50000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr-high", type=float, default=3e-4)
    parser.add_argument("--lr-low", type=float, default=3e-4)
    parser.add_argument("--lr-irm", type=float, default=3e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--trajectory-len", type=int, default=512)
    parser.add_argument("--high-level-period", type=int, default=50)
    parser.add_argument("--eta-irm", type=float, default=0.01)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--loss-irm-weight", type=float, default=1.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DummyDRLMTUCSEnvironment()
    agent = HierarchicalPPOAgent(args, env)

    print("=== DRL-MTUCS Hierarchical PPO Training Started ===")
    total_steps = 0
    update_id = 0

    while total_steps < args.total_timesteps:

        obs, act, logp, vals, rets, adv, low_buf = agent.collect_trajectory()

        irm_loss, lp, lv = agent.update_low_and_irm(low_buf)

        p_loss, v_loss = agent.update_high(obs, act, logp, vals, rets, adv)

        total_steps += len(act)
        update_id += 1

        print(f"\n--- Update {update_id} ---")
        print(f"Steps     : {total_steps}")
        print(f"High PPO  : Policy={p_loss:.4f}, Value={v_loss:.4f}")
        print(f"IRM       : Loss={irm_loss:.4f}")

    print("=== Training Finished ===")


if __name__ == "__main__":
    main()
