import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
from collections import deque
from typing import List, Tuple

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
from envs.real_mtucs_env import RealMTUCSEnvironment


# ============================================================
# 分层 PPO 代理
# ============================================================
class HierarchicalPPOAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ← 添加缺失的属性
        self.allocation_interval = args.high_level_period  # 高层决策间隔（50 步）
        
        # === 修复：使用正确的维度 ===
        uav_dim = 6   # pos(2) + vel(2) + energy(1) + task_id(1)
        task_dim = 5  # pos(2) + urgency(1) + time(1) + progress(1)
        
        self.high_allocator = MultiHeadHighLevelAllocator(
            uav_dim=uav_dim,      # ← 改为 6
            task_dim=task_dim,    # ← 改为 5
            embed_dim=64,
            num_heads=2,
            hidden_dim=128
        ).to(self.device)
        
        # ← 添加高层价值网络
        self.high_value_net = nn.Sequential(
            nn.Linear(env.global_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # --- 内在奖励模块 ---
        self.intrinsic_reward_module = IntrinsicRewardModule(
            obs_dim=env.low_obs_dim,
            action_dim=env.low_action_dim,
            feature_dim=args.feature_dim
        ).to(self.device)

        # --- 低层策略（使用启发式策略） ---
        self.low_policy = SimpleHeuristicLowPolicy(env.low_obs_dim, env.low_action_dim).to(self.device)

        # --- 优化器 ---
        self.optimizer_high = optim.Adam(
            list(self.high_allocator.parameters()) + list(self.high_value_net.parameters()),
            lr=args.lr_high, 
            eps=args.eps
        )
        
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
        """收集一整个轨迹"""
        global_obs, low_obs_list = self.env.reset()
        
        # ← 确保观测在正确的设备上
        global_obs = global_obs.to(self.device)
        low_obs_list = [obs.to(self.device) for obs in low_obs_list]
        
        trajectory_data = []
        low_buffer = []
        
        for step in range(self.args.trajectory_len):
            # === 1. 高层决策 ===
            if step % self.allocation_interval == 0:
                # 拆分 global_obs 为 uav_feat 和 task_feat
                uav_feat, task_feat = self._split_global_obs(global_obs)
                
                # 获取 logits: (H, U, T) 或 (B, H, U, T)
                # 使用 torch.no_grad() 避免在收集阶段构建不必要的计算图
                with torch.no_grad():
                    logits = self.high_allocator(uav_feat, task_feat)
                    
                    # 确保维度是 (H, U, T)
                    if logits.dim() == 4:
                        logits = logits.squeeze(0)
                    
                    # 聚合逻辑：Softmax -> Mean(Heads) -> Mean(UAVs)
                    probs = F.softmax(logits, dim=-1) # (H, U, T)
                    probs = probs.mean(dim=0)         # (U, T)
                    high_action_probs = probs.mean(dim=0) # (T)
                    
                    high_dist = torch.distributions.Categorical(high_action_probs)
                    high_action = high_dist.sample()  # 标量
                    high_log_prob = high_dist.log_prob(high_action)
                    
                    # 计算 Value (也需要 detach)
                    high_value = self.high_value_net(global_obs).squeeze()

                # 应用高层决策
                high_action_np = high_action.cpu().numpy()
                
                # 将单个任务ID扩展为每个UAV分配不同任务 (简单的循环分配策略)
                base_task = int(high_action_np.item()) if high_action_np.ndim == 0 else int(high_action_np[0])
                allocation = [(base_task + i) % self.env.num_tasks for i in range(self.env.num_uavs)]
                
                self.env.apply_high_level_allocation(allocation)
                
                # 记录样本 (关键修复：添加 .detach())
                sample = {
                    'global_obs': global_obs.clone().detach(), # ← detach
                    'uav_feat': uav_feat.clone().detach(),     # ← detach
                    'task_feat': task_feat.clone().detach(),   # ← detach
                    'high_action': high_action.clone().detach(), # ← detach
                    'high_log_prob': high_log_prob.clone().detach(), # ← detach
                    'high_value': high_value.clone().detach(), # ← detach
                    'reward': 0.0,
                    'done': False
                }
                trajectory_data.append(sample)
            
            # === 2. 低层执行 ===
            low_actions = []
            for i, obs in enumerate(low_obs_list):
                # 低层策略通常不需要梯度（如果是启发式），或者在 update_low 中处理
                with torch.no_grad():
                    action = self.low_policy(obs.unsqueeze(0))
                action_np = action.squeeze(0).detach().cpu().numpy()
                low_actions.append(action_np)
            
            # === 3. 环境交互 ===
            next_global_obs, next_low_obs, reward, done = self.env.step(low_actions)
            
            next_global_obs = next_global_obs.to(self.device)
            next_low_obs = [obs.to(self.device) for obs in next_low_obs]
            
            for i, obs in enumerate(low_obs_list):
                low_buffer.append({
                    'obs': obs.clone().detach(), # ← detach
                    'action': torch.from_numpy(low_actions[i]).float().to(self.device),
                    'next_obs': next_low_obs[i].clone().detach() # ← detach
                })
            
            if len(trajectory_data) > 0:
                trajectory_data[-1]['reward'] += reward
                trajectory_data[-1]['done'] = done
            
            global_obs = next_global_obs
            low_obs_list = next_low_obs
            
            if done:
                global_obs, low_obs_list = self.env.reset()
                global_obs = global_obs.to(self.device)
                low_obs_list = [obs.to(self.device) for obs in low_obs_list]
                print(f"  [DEBUG] Episode ended at step {step}, restarting...")
        
        print(f"  [INFO] Collected {len(trajectory_data)} high-level samples")
        
        # === 4. 计算 GAE 和 Returns ===
        if len(trajectory_data) == 0:
            empty = torch.tensor([]).to(self.device)
            return empty, empty, empty, empty, empty, empty, empty, empty, []
        
        obs_list = torch.stack([d['global_obs'] for d in trajectory_data])
        uav_feats = torch.stack([d['uav_feat'] for d in trajectory_data])
        task_feats = torch.stack([d['task_feat'] for d in trajectory_data])
        actions = torch.stack([d['high_action'] for d in trajectory_data])
        log_probs = torch.stack([d['high_log_prob'] for d in trajectory_data])
        values = torch.stack([d['high_value'] for d in trajectory_data])
        rewards = [d['reward'] for d in trajectory_data]
        dones = [d['done'] for d in trajectory_data]
        
        # 计算最后一个状态的价值（用于 GAE）
        with torch.no_grad():
            final_value = self.high_value_net(global_obs).squeeze()
        
        returns, advantages = self._compute_gae_and_returns(rewards, values, dones, final_value)
        
        return obs_list, uav_feats, task_feats, actions, log_probs, values, returns, advantages, low_buffer

    def _compute_gae_and_returns(self, rewards, values, dones, final_value):
        """计算 GAE 和 Returns"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = final_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages).float().to(self.device)
        returns = advantages + values
        
        # 添加 Return 归一化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # ← 新增
    
        return returns, advantages

    def update_high(self, obs, uav_feats, task_feats, actions, old_log_probs, values, returns, advantages):
        """更新高层策略"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 维度检查：actions 应该是 [batch_size]
        if actions.dim() > 1:
            actions = actions.squeeze()
            
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.ppo_epochs):
            # forward 返回 (B, H, U, T)
            logits = self.high_allocator(uav_feats, task_feats) 
            
            # 聚合逻辑必须与 collect_trajectory 一致
            probs = F.softmax(logits, dim=-1) # (B, H, U, T)
            probs = probs.mean(dim=1)         # (B, U, T)
            action_probs = probs.mean(dim=1)  # (B, T)
            
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            new_values = self.high_value_net(obs).squeeze(-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - new_values).pow(2).mean()
            
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer_high.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.high_allocator.parameters()) + list(self.high_value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer_high.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss / self.ppo_epochs, total_value_loss / self.ppo_epochs

    def update_low_and_irm(self, low_buffer):
        """更新内在奖励模块"""
        if len(low_buffer) == 0:
            return 0.0, 0.0, 0.0
        
        total_fwd_loss = 0
        total_inv_loss = 0
        
        for data in low_buffer:
            obs = data["obs"].to(self.device)
            action = data["action"].to(self.device)
            next_obs = data["next_obs"].to(self.device)
            
            _, loss_fwd, loss_inv = self.intrinsic_reward_module(obs, action, next_obs)
            
            loss = self.args.loss_irm_weight * (loss_fwd + loss_inv)
            
            self.optimizer_irm.zero_grad()
            loss.backward()
            self.optimizer_irm.step()
            
            total_fwd_loss += loss_fwd.item()
            total_inv_loss += loss_inv.item()
        
        n = len(low_buffer)
        return total_fwd_loss / n, total_inv_loss / n, (total_fwd_loss + total_inv_loss) / n

    def _split_global_obs(self, global_obs):
        """
        拆分 global_obs 为 uav_feat 和 task_feat
        
        global_obs 结构 (from real_mtucs_env.py):
        - [0:30] UAV 状态 (5 UAV * 6 维)
        - [30:55] Task 状态 (5 Task * 5 维)
        - [55:65] 全局统计
        
        Args:
            global_obs: [global_obs_dim] 或 [batch, global_obs_dim]
        
        Returns:
            uav_feat: [num_uavs, uav_dim] 或 [batch, num_uavs, uav_dim]
            task_feat: [num_tasks, task_dim] 或 [batch, num_tasks, task_dim]
        """
        num_uavs = self.env.num_uavs  # 5
        num_tasks = self.env.num_tasks  # 5
        uav_dim = 6  # pos(2) + vel(2) + energy(1) + task_id(1)
        task_dim = 5  # pos(2) + urgency(1) + time(1) + progress(1)
        
        # ← 确保 global_obs 在正确的设备上
        if not global_obs.is_cuda:
            global_obs = global_obs.to(self.device)
        
        if global_obs.dim() == 1:
            # 单个样本: [global_obs_dim]
            uav_feat = global_obs[:num_uavs * uav_dim].view(num_uavs, uav_dim)
            task_feat = global_obs[num_uavs * uav_dim : num_uavs * uav_dim + num_tasks * task_dim].view(num_tasks, task_dim)
        else:
            # 批量样本: [batch, global_obs_dim]
            batch_size = global_obs.shape[0]
            uav_feat = global_obs[:, :num_uavs * uav_dim].view(batch_size, num_uavs, uav_dim)
            task_feat = global_obs[:, num_uavs * uav_dim : num_uavs * uav_dim + num_tasks * task_dim].view(batch_size, num_tasks, task_dim)
        
        return uav_feat, task_feat


# ============================================================
# 简单启发式低层策略
# ============================================================
class SimpleHeuristicLowPolicy(nn.Module):
    """简单的启发式低层策略：直接朝目标飞"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
    def forward(self, obs):
        """
        obs 结构:
        [0-5]: 自身状态 (pos_x, pos_y, vel_x, vel_y, energy, task_id)
        [6-10]: 最近任务1 (rel_x, rel_y, dist, urgency, time)
        """
        # 提取最近任务的相对位置 (索引 6-7)
        if obs.dim() == 1:
            relative_vec = obs[6:8]
        else:
            relative_vec = obs[:, 6:8]
        
        # 归一化方向
        norm = torch.norm(relative_vec, dim=-1, keepdim=True) + 1e-8
        direction = relative_vec / norm
        
        # 输出动作: [vx, vy, task_select, hover]
        if obs.dim() == 1:
            action = torch.zeros(self.action_dim)
            action[0] = direction[0]
            action[1] = direction[1]
        else:
            action = torch.zeros(obs.shape[0], self.action_dim)
            action[:, 0] = direction[:, 0]
            action[:, 1] = direction[:, 1]
        
        return action


# ============================================================
# 主训练函数
# ============================================================
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

    parser.add_argument("--trajectory-len", type=int, default=512)  # ← 保持 512
    parser.add_argument("--high-level-period", type=int, default=50)
    parser.add_argument("--eta-irm", type=float, default=0.01)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--loss-irm-weight", type=float, default=1.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # env = DummyDRLMTUCSEnvironment()
    env = RealMTUCSEnvironment(
        num_uavs=5,
        num_tasks=5,       # ← 改为 5
        map_size=200.0,    # ← 改为 200
        max_steps=200,     # ← 改为 200
        uav_speed=50.0,    # ← 改为 50
        dt=1.0
    )
    
    agent = HierarchicalPPOAgent(args, env)

    print("=== DRL-MTUCS Hierarchical PPO Training Started ===")
    total_steps = 0
    update_id = 0

    # 在 main() 中补全训练循环
    while total_steps < args.total_timesteps:
        # ← 修复：接收 9 个返回值（增加了 uav_feats 和 task_feats）
        obs, uav_feats, task_feats, act, logp, vals, rets, adv, low_buf = agent.collect_trajectory()
        
        if len(act) == 0:
            continue
        
        irm_loss, _, _ = agent.update_low_and_irm(low_buf)
        
        # ← 修复：传递 uav_feats 和 task_feats
        p_loss, v_loss = agent.update_high(obs, uav_feats, task_feats, act, logp, vals, rets, adv)
        
        total_steps += len(act)
        update_id += 1
        
        if update_id % 10 == 0:
            print(f"Update {update_id} | Steps {total_steps} | Policy Loss: {p_loss:.4f} | Value Loss: {v_loss:.4f} | IRM Loss: {irm_loss:.4f}")
        
        # 保存模型
        if update_id % 100 == 0:
            torch.save({
                'high': agent.high_allocator.state_dict(),
                'low': agent.low_policy.state_dict(),
                'irm': agent.intrinsic_reward_module.state_dict()
            }, f"checkpoint_{update_id}.pth")

    print("=== Training Finished ===")


if __name__ == "__main__":
    main()
