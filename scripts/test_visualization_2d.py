import torch
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 路径设置 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.train import HierarchicalPPOAgent

# ============================================================
# 2D 可视化仿真环境 (Mock)
# ============================================================
class Mock2DEnvironment:
    def __init__(self, num_uavs=5, num_tasks=10):
        self.num_uavs = num_uavs
        self.num_tasks = num_tasks
        self.map_size = 200.0
        
        # 维度定义
        self.global_obs_dim = 540 
        self.low_obs_dim = 42 
        self.low_action_dim = 2 
        
        # 内部状态
        self.uav_pos = np.zeros((num_uavs, 2))
        self.task_pos = np.zeros((num_tasks, 2))
        self.task_urgency = np.zeros(num_tasks)
        self.uav_assignments = [-1] * num_uavs

    def reset(self):
        self.uav_pos = np.random.rand(self.num_uavs, 2) * self.map_size
        self.task_pos = np.random.rand(self.num_tasks, 2) * self.map_size
        self.task_urgency = np.random.rand(self.num_tasks)
        return self._get_global_obs(), self._get_low_obs()

    def _get_global_obs(self):
        uav_feats = []
        for i in range(self.num_uavs):
            feat = np.concatenate([self.uav_pos[i], [0, 0], [1.0], [self.uav_assignments[i]]])
            uav_feats.append(feat)
        uav_part = np.concatenate(uav_feats)
        
        task_feats = []
        for i in range(100): 
            if i < self.num_tasks:
                feat = np.concatenate([self.task_pos[i], [self.task_urgency[i]], [0], [0]])
            else:
                feat = np.zeros(5)
            task_feats.append(feat)
        task_part = np.concatenate(task_feats)
        
        global_part = np.zeros(10)
        obs = np.concatenate([uav_part, task_part, global_part])
        return torch.from_numpy(obs).float()

    def _get_low_obs(self):
        low_obs_list = []
        for i in range(self.num_uavs):
            obs = np.zeros(self.low_obs_dim)
            obs[0:2] = self.uav_pos[i]
            target_idx = self.uav_assignments[i]
            if target_idx != -1 and target_idx < self.num_tasks:
                obs[2:4] = self.task_pos[target_idx] - self.uav_pos[i]
            low_obs_list.append(torch.from_numpy(obs).float())
        return low_obs_list

    def step(self, low_actions):
        speed_factor = 5.0 # 增加速度，确保肉眼可见
        
        for i, action in enumerate(low_actions):
            # action 是归一化的方向向量
            dx = np.clip(action[0], -1, 1) * speed_factor
            dy = np.clip(action[1], -1, 1) * speed_factor
            self.uav_pos[i, 0] += dx
            self.uav_pos[i, 1] += dy
            self.uav_pos[i] = np.clip(self.uav_pos[i], 0, self.map_size)

        # 任务完成逻辑
        for i in range(self.num_uavs):
            target_idx = self.uav_assignments[i]
            if target_idx != -1 and target_idx < self.num_tasks:
                dist = np.linalg.norm(self.uav_pos[i] - self.task_pos[target_idx])
                if dist < 5.0: # 到达判定距离
                    self.task_urgency[target_idx] = max(0, self.task_urgency[target_idx] - 0.2)
                    if self.task_urgency[target_idx] <= 0.1:
                        # 任务重生
                        self.task_pos[target_idx] = np.random.rand(2) * self.map_size
                        self.task_urgency[target_idx] = np.random.rand()

        return self._get_global_obs(), self._get_low_obs(), 0.0, False

    def apply_high_level_allocation(self, allocation):
        if isinstance(allocation, (int, float, np.integer)):
            base = int(allocation)
            for i in range(self.num_uavs):
                self.uav_assignments[i] = (base + i) % self.num_tasks
        elif isinstance(allocation, list):
            for i, task_idx in enumerate(allocation):
                if i < self.num_uavs:
                    self.uav_assignments[i] = int(task_idx) % self.num_tasks

# ============================================================
# 动画可视化主逻辑
# ============================================================
def run_2d_visualization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint")
    # ... 其他参数 ...
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--high-level-period", type=int, default=20)
    parser.add_argument("--eta-irm", type=float, default=0.01)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--loss-irm-weight", type=float, default=1.0)
    
    args = parser.parse_args()

    env = Mock2DEnvironment(num_uavs=5, num_tasks=10)
    agent = HierarchicalPPOAgent(args, env)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=agent.device)
        if 'high' in checkpoint: agent.high_allocator.load_state_dict(checkpoint['high'])
        # 低层策略我们这里不用了，所以不加载也没关系
    else:
        print("Running with random weights.")

    global_obs, low_obs_list = env.reset()
    global_obs = global_obs.to(agent.device)
    
    # --- Matplotlib 动画设置 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_title("Multi-UAV Task Allocation (DRL)")
    ax.grid(True, linestyle='--', alpha=0.5)

    # 绘图元素
    # 任务点：蓝色散点，大小随紧急度变化
    task_scat = ax.scatter([], [], c='blue', s=[], alpha=0.6, label='Tasks')
    # 无人机：红色点
    uav_scat = ax.scatter([], [], c='red', s=100, marker='^', label='UAVs')
    # 连线：UAV -> Task
    lines = [ax.plot([], [], 'k--', alpha=0.3)[0] for _ in range(env.num_uavs)]
    # 文本标签
    uav_texts = [ax.text(0, 0, '', fontsize=8) for _ in range(env.num_uavs)]

    step_count = [0] # 使用列表以便在闭包中修改

    def update(frame):
        nonlocal global_obs, low_obs_list
        
        # 1. 高层决策 (使用训练好的模型)
        if step_count[0] % args.high_level_period == 0:
            uav_feat, task_feat = agent._split_global_obs(global_obs)
            with torch.no_grad():
                logits = agent.high_allocator(uav_feat, task_feat)
                if logits.dim() == 4: logits = logits.squeeze(0)
                probs = torch.softmax(logits, dim=-1).mean(dim=0).mean(dim=0)
                high_action = torch.argmax(probs).item()
            
            allocation = [(high_action + i) % env.num_tasks for i in range(env.num_uavs)]
            env.apply_high_level_allocation(allocation)

        # 2. 低层执行 (=== 修改处：使用规则导航，确保能动 ===)
        low_actions = []
        for i in range(env.num_uavs):
            # 获取当前目标
            target_idx = env.uav_assignments[i]
            
            if target_idx != -1 and target_idx < env.num_tasks:
                # 计算方向向量
                diff = env.task_pos[target_idx] - env.uav_pos[i]
                dist = np.linalg.norm(diff)
                if dist > 0.1:
                    action = diff / dist  # 归一化方向
                else:
                    action = np.zeros(2)
            else:
                action = np.zeros(2) # 没有任务时不动
                
            low_actions.append(action)

        # 3. 环境步进
        next_global_obs, next_low_obs_list, _, _ = env.step(low_actions)
        global_obs = next_global_obs.to(agent.device)
        low_obs_list = next_low_obs_list
        step_count[0] += 1

        # 4. 更新图形
        # 更新任务
        task_scat.set_offsets(env.task_pos)
        task_scat.set_sizes(env.task_urgency * 300 + 50) # 紧急度越大，点越大

        # 更新无人机
        uav_scat.set_offsets(env.uav_pos)

        # 更新连线和标签
        for i in range(env.num_uavs):
            uav_pos = env.uav_pos[i]
            target_idx = env.uav_assignments[i]
            
            # 更新连线
            if target_idx != -1 and target_idx < env.num_tasks:
                target_pos = env.task_pos[target_idx]
                lines[i].set_data([uav_pos[0], target_pos[0]], [uav_pos[1], target_pos[1]])
            else:
                lines[i].set_data([], [])
            
            # 更新标签
            uav_texts[i].set_position((uav_pos[0] + 2, uav_pos[1] + 2))
            uav_texts[i].set_text(f"U{i}->T{target_idx}")

        ax.set_title(f"Step: {step_count[0]}")
        return [task_scat, uav_scat] + lines + uav_texts

    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run_2d_visualization()