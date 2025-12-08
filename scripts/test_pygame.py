import torch
import numpy as np
import argparse
import sys
import os
import time

# --- 尝试导入 PyGame ---
try:
    import pygame
except ImportError:
    print("错误: 未安装 pygame 库。")
    print("请运行: pip install pygame")
    sys.exit(1)

# --- 路径设置 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.train import HierarchicalPPOAgent

# ============================================================
# 2D 可视化仿真环境 (Mock) - 保持不变
# ============================================================
class Mock2DEnvironment:
    def __init__(self, num_uavs=5, num_tasks=10):
        self.num_uavs = num_uavs
        self.num_tasks = num_tasks
        self.map_size = 200.0
        
        self.global_obs_dim = 540 
        self.low_obs_dim = 42 
        self.low_action_dim = 2 
        
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
        speed_factor = 2.0 # 稍微降低速度，PyGame 帧率高，看起来会很顺滑
        
        for i, action in enumerate(low_actions):
            dx = np.clip(action[0], -1, 1) * speed_factor
            dy = np.clip(action[1], -1, 1) * speed_factor
            self.uav_pos[i, 0] += dx
            self.uav_pos[i, 1] += dy
            self.uav_pos[i] = np.clip(self.uav_pos[i], 0, self.map_size)

        for i in range(self.num_uavs):
            target_idx = self.uav_assignments[i]
            if target_idx != -1 and target_idx < self.num_tasks:
                dist = np.linalg.norm(self.uav_pos[i] - self.task_pos[target_idx])
                if dist < 5.0:
                    self.task_urgency[target_idx] = max(0, self.task_urgency[target_idx] - 0.2)
                    if self.task_urgency[target_idx] <= 0.1:
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
# PyGame 可视化主逻辑
# ============================================================
def run_pygame_visualization():
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

    # 初始化环境和代理
    env = Mock2DEnvironment(num_uavs=5, num_tasks=10)
    agent = HierarchicalPPOAgent(args, env)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=agent.device)
        if 'high' in checkpoint: agent.high_allocator.load_state_dict(checkpoint['high'])
    else:
        print("Running with random weights.")

    global_obs, low_obs_list = env.reset()
    global_obs = global_obs.to(agent.device)

    # --- PyGame 设置 ---
    pygame.init()
    
    # 窗口尺寸
    WINDOW_SIZE = 800
    SCALE = WINDOW_SIZE / env.map_size
    
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("DRL-MTUCS Visualization (PyGame) - Press SPACE to Pause")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    large_font = pygame.font.SysFont("Arial", 32, bold=True)

    # 颜色定义
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    RED = (200, 50, 50) # 默认暂停颜色
    
    # 为每个 UAV 定义独特的颜色 (RGB)
    UAV_COLORS = [
        (255, 0, 0),    # UAV 0: Red
        (0, 200, 0),    # UAV 1: Green
        (0, 0, 255),    # UAV 2: Blue
        (255, 165, 0),  # UAV 3: Orange
        (128, 0, 128),  # UAV 4: Purple
        (0, 255, 255),  # Cyan (备用)
        (255, 0, 255)   # Magenta (备用)
    ]
    DEFAULT_TASK_COLOR = (150, 150, 150) # 未分配任务的颜色

    running = True
    paused = False
    step_count = 0

    while running:
        # 1. 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # 2. 高层决策
            if step_count % args.high_level_period == 0:
                uav_feat, task_feat = agent._split_global_obs(global_obs)
                with torch.no_grad():
                    logits = agent.high_allocator(uav_feat, task_feat)
                    if logits.dim() == 4: logits = logits.squeeze(0)
                    probs = torch.softmax(logits, dim=-1).mean(dim=0).mean(dim=0)
                    high_action = torch.argmax(probs).item()
                
                allocation = [(high_action + i) % env.num_tasks for i in range(env.num_uavs)]
                env.apply_high_level_allocation(allocation)

            # 3. 低层执行
            low_actions = []
            for i in range(env.num_uavs):
                target_idx = env.uav_assignments[i]
                if target_idx != -1 and target_idx < env.num_tasks:
                    diff = env.task_pos[target_idx] - env.uav_pos[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0.5:
                        action = diff / dist
                    else:
                        action = np.zeros(2)
                else:
                    action = np.zeros(2)
                low_actions.append(action)

            # 4. 环境步进
            next_global_obs, next_low_obs_list, _, _ = env.step(low_actions)
            global_obs = next_global_obs.to(agent.device)
            low_obs_list = next_low_obs_list
            step_count += 1

        # 5. 渲染绘制
        screen.fill(WHITE)

        # 绘制网格
        for x in range(0, WINDOW_SIZE, 50):
            pygame.draw.line(screen, (240, 240, 240), (x, 0), (x, WINDOW_SIZE))
            pygame.draw.line(screen, (240, 240, 240), (0, x), (WINDOW_SIZE, x))

        # 绘制连线 (UAV -> Task)
        for i in range(env.num_uavs):
            target_idx = env.uav_assignments[i]
            if target_idx != -1 and target_idx < env.num_tasks:
                start_pos = env.uav_pos[i] * SCALE
                end_pos = env.task_pos[target_idx] * SCALE
                # 连线颜色与 UAV 颜色一致
                color = UAV_COLORS[i % len(UAV_COLORS)]
                pygame.draw.line(screen, color, start_pos, end_pos, 2)

        # 绘制任务
        for i in range(env.num_tasks):
            pos = (int(env.task_pos[i][0] * SCALE), int(env.task_pos[i][1] * SCALE))
            urgency = env.task_urgency[i]
            radius = int(10 + urgency * 15)
            
            # 确定任务颜色
            task_color = DEFAULT_TASK_COLOR
            # 检查该任务是否被分配给了某个 UAV
            assigned_uav = -1
            for u_idx, t_idx in enumerate(env.uav_assignments):
                if t_idx == i:
                    assigned_uav = u_idx
                    break
            
            if assigned_uav != -1:
                task_color = UAV_COLORS[assigned_uav % len(UAV_COLORS)]
            
            pygame.draw.circle(screen, task_color, pos, radius)
            # 绘制任务ID (白色文字)
            text = font.render(f"T{i}", True, WHITE)
            screen.blit(text, (pos[0]-8, pos[1]-8))

        # 绘制无人机
        for i in range(env.num_uavs):
            pos = (int(env.uav_pos[i][0] * SCALE), int(env.uav_pos[i][1] * SCALE))
            # 无人机颜色
            uav_color = UAV_COLORS[i % len(UAV_COLORS)]
            pygame.draw.circle(screen, uav_color, pos, 12)
            # 绘制 UAV ID
            text = font.render(f"U{i}", True, WHITE)
            screen.blit(text, (pos[0]-8, pos[1]-8))

        # 绘制信息面板
        info_text = font.render(f"Step: {step_count} | High-Level Period: {args.high_level_period} | SPACE to Pause", True, BLACK)
        screen.blit(info_text, (10, 10))

        if paused:
            pause_text = large_font.render("PAUSED", True, RED)
            text_rect = pause_text.get_rect(center=(WINDOW_SIZE/2, 50))
            screen.blit(pause_text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_pygame_visualization()