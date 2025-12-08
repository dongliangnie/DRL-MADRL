import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# --- 确保可以导入项目模块 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 注意：这里需要导入真实的 Environment，因为 Dummy 环境可能无法正确反映训练效果
# 如果你想用 Dummy 环境测试代码流程，可以改回 DummyDRLMTUCSEnvironment
from envs.real_mtucs_env import RealMTUCSEnvironment
from scripts.train import HierarchicalPPOAgent

def run_visualization_test():
    # 1. 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=5000) # 测试步数
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
    parser.add_argument("--trajectory-len", type=int, default=256) 
    parser.add_argument("--high-level-period", type=int, default=50) 
    parser.add_argument("--eta-irm", type=float, default=0.01)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--loss-irm-weight", type=float, default=1.0)
    args = parser.parse_args()

    # 2. 初始化环境和代理
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 使用真实环境进行更有意义的可视化
    env = RealMTUCSEnvironment(num_uavs=5, num_tasks=5) 
    agent = HierarchicalPPOAgent(args, env)

    # === 加载权重 ===
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=agent.device)
        
        # 加载高层策略
        if 'high' in checkpoint:
            try:
                agent.high_allocator.load_state_dict(checkpoint['high'])
                print("High-level allocator loaded.")
            except Exception as e:
                print(f"Warning: Failed to load high-level allocator: {e}")
        
        # 加载低层策略
        if 'low' in checkpoint:
            try:
                agent.low_policy.load_state_dict(checkpoint['low'])
                print("Low-level policy loaded.")
            except Exception as e:
                print(f"Warning: Failed to load low-level policy: {e}")
                
        # 加载 IRM (可选，仅用于继续训练，推理时不一定需要)
        if 'irm' in checkpoint:
            try:
                agent.intrinsic_reward_module.load_state_dict(checkpoint['irm'])
                print("IRM loaded.")
            except Exception as e:
                print(f"Warning: Failed to load IRM: {e}")
    else:
        print("No checkpoint provided or file not found. Running with random weights (Untrained).")

    # 3. 准备数据存储容器
    history = {
        "steps": [],
        "avg_reward": [],
        "value_estimate": []
    }

    # 4. 设置 Matplotlib 实时绘图
    plt.ion() # 开启交互模式
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Evaluation: {os.path.basename(args.checkpoint) if args.checkpoint else "Random Agent"}', fontsize=16)
    
    ax_reward = axs[0]
    ax_value = axs[1]

    total_steps = 0
    update_id = 0

    print("=== 开始可视化评估 ===")
    print("注意：这将弹出一个窗口实时显示评估曲线。")

    # 设置为评估模式 (虽然 PPO 主要是 sample，但有些层可能有 dropout/batchnorm)
    agent.high_allocator.eval()
    agent.high_value_net.eval()
    agent.low_policy.eval()

    while total_steps < args.total_timesteps:
        # --- 收集轨迹 (不进行更新) ---
        # 注意：collect_trajectory 内部已经包含了环境交互逻辑
        with torch.no_grad():
            # 这里我们只关心返回的 rewards 和 values
            _, _, _, _, _, vals, rets, _, _ = agent.collect_trajectory()
        
        # 计算统计数据
        current_avg_reward = torch.tensor(rets).mean().item() if len(rets) > 0 else 0.0
        current_avg_value = vals.mean().item() if len(vals) > 0 else 0.0
        
        # 这里的 steps 是高层决策次数
        steps_in_batch = len(rets)
        total_steps += steps_in_batch
        update_id += 1

        if steps_in_batch == 0:
            continue

        # --- 记录数据 ---
        history["steps"].append(total_steps)
        history["avg_reward"].append(current_avg_reward)
        history["value_estimate"].append(current_avg_value)

        # --- 实时绘图更新 ---
        for ax in axs:
            ax.cla()

        # 绘制 Average Reward
        ax_reward.plot(history["steps"], history["avg_reward"], 'm-o', label='Avg Return')
        ax_reward.set_title('Average Return per Trajectory')
        ax_reward.set_ylabel('Return')
        ax_reward.grid(True)
        ax_reward.legend()

        # 绘制 Value Estimate
        ax_value.plot(history["steps"], history["value_estimate"], 'b-o', label='Value Est.')
        ax_value.set_title('Average Value Estimate')
        ax_value.set_xlabel('High-Level Decision Steps')
        ax_value.set_ylabel('Value')
        ax_value.grid(True)
        ax_value.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1) 

        print(f"Batch {update_id}: Steps={total_steps}, Avg Reward={current_avg_reward:.2f}, Avg Value={current_avg_value:.2f}")

    plt.ioff() 
    print("=== 评估结束 ===")
    plt.show() 

if __name__ == "__main__":
    run_visualization_test()