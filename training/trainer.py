"""
Train Script for DRL-MTUCS
--------------------------
集成：
- 环境初始化
- High-Level Allocator
- Low-Level UAV Policy
- Temporal Predictor
- DRL-MTUCS 主类
- PPO 训练循环

运行方式：
    python scripts/train.py

可选参数：
    python scripts/train.py --config config/config.yaml --episodes 2000
"""

import os
import sys
import yaml
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# 将项目根目录加入 Python 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# === 导入项目内部模块 ===
from environment.uav_env import UAVEnvironment
from models.high_level_allocator import MultiHeadHighLevelAllocator
from models.low_level_uav_policy import LowLevelUAVPolicy
from models.temporal_predictor import TemporalPredictor
from core.drl_mtucs import DRL_MTUCS
from training.ppo import PPOTrainer


# ============================================================
# 解析命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train DRL-MTUCS")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--env_config", type=str, default="config/environment.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


# ============================================================
# 加载 YAML 配置
# ============================================================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 主训练函数
# ============================================================

def train():
    args = parse_args()

    # 1. 加载配置文件
    print("Loading configuration ...")
    cfg = load_yaml(args.config)
    env_cfg = load_yaml(args.env_config)

    # 2. 创建 log / checkpoint 目录
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # 3. 初始化环境
    print("Initializing UAV environment ...")
    env = UAVEnvironment(env_cfg)

    # 获取环境维度信息
    num_uav = env.num_uav
    uav_obs_dim = env.uav_obs_dim
    task_obs_dim = env.task_obs_dim

    # 4. 初始化模型组件
    print("Initializing model components ...")

    # --- 高层分配器 ---
    high_level = MultiHeadHighLevelAllocator(
        uav_dim=uav_obs_dim,
        task_dim=task_obs_dim,
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"]
    )

    # --- 低层 UAV 策略 ---
    low_level = LowLevelUAVPolicy(
        obs_dim=uav_obs_dim,
        action_dim=env.action_dim,
        hidden_dim=256
    )

    # --- 时间预测器 ---
    temporal_predictor = TemporalPredictor(
        obs_dim=uav_obs_dim,
        goal_dim=uav_obs_dim,
        embedding_dim=128
    )

    # 5. 创建 DRL-MTUCS 主类
    drl = DRL_MTUCS(
        env=env,
        high_level=high_level,
        low_level=low_level,
        temporal_predictor=temporal_predictor,
        config=cfg
    )

    # 6. 创建 PPO 训练器
    ppo = PPOTrainer(
        high_level_policy=high_level,
        low_level_policy=low_level,
        config=cfg["ppo"]
    )

    # 7. 训练轮数
    total_episodes = args.episodes or cfg["train"]["episodes"]

    print(f"\n===== Start Training ({total_episodes} episodes) =====\n")

    # ============================================================
    # 主训练循环
    # ============================================================

    for ep in range(1, total_episodes + 1):

        # 执行一整条 episode
        ep_reward, ep_metrics = drl.run_episode(ppo=ppo)

        # Tensorboard logging
        writer.add_scalar("Episode/Reward", ep_reward, ep)
        for k, v in ep_metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, ep)

        print(f"[Episode {ep}] Reward = {ep_reward:.2f}")

        # 每隔 N 轮保存一次模型
        if ep % cfg["train"]["save_interval"] == 0:
            save_path = os.path.join(args.save_dir, f"episode_{ep}")
            os.makedirs(save_path, exist_ok=True)

            torch.save(high_level.state_dict(), os.path.join(save_path, "high_level.pt"))
            torch.save(low_level.state_dict(), os.path.join(save_path, "low_level.pt"))
            torch.save(temporal_predictor.state_dict(), os.path.join(save_path, "temporal_predictor.pt"))

            print(f"Checkpoint saved at episode {ep}")

    writer.close()
    print("\n===== Training Completed! =====")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    train()
