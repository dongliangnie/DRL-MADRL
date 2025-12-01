# 项目结构
DRL-MTUCS/
│
├── config/
│   ├── __init__.py
│   ├── config.yaml          # 主要配置文件，包含所有超参数
│   └── environment.yaml     # 环境相关的配置（如地图大小、UAV参数等）
│
├── environment/
│   ├── __init__.py
│   ├── uav_env.py           # 主环境类，模拟UAV和任务
│   ├── task_generator.py    # 生成监控和紧急任务
│   ├── aoi_manager.py       # 管理所有任务的AoI
│   └── communication_model.py # 通信模型和能量消耗模型
│
├── models/
│   ├── __init__.py
│   ├── high_level_allocator.py   # 高层分配器网络
│   ├── low_level_uav_policy.py   # 低层UAV策略网络
│   ├── temporal_predictor.py     # 时间预测器网络
│   └── dynamic_weighted_queue.py # 动态加权队列实现
│
├── core/
│   ├── __init__.py
│   ├── drl_mtucs.py         # DRL-MTUCS算法主类，整合所有组件
│   ├── intrinsic_reward.py  # 自平衡内在奖励计算
│   └── experience_replay.py # 经验回放缓冲区（如果需要）
│
├── training/
│   ├── __init__.py
│   ├── trainer.py           # 训练循环
│   ├── ppo.py               # PPO算法实现（或其他actor-critic算法）
│   └── utils.py             # 训练工具函数（如GAE计算）
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py         # 评估训练好的模型
│   └── visualizer.py        # 可视化UAV轨迹和性能指标
│
├── data/                    # 存储数据集（如San Francisco和Chengdu的轨迹数据）
│   ├── san_francisco/
│   └── chengdu/
│
├── scripts/
│   ├── train.py             # 启动训练脚本
│   ├── evaluate.py          # 启动评估脚本
│   └── visualize.py         # 启动可视化脚本
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # 加载和处理真实数据集
│   ├── reward_calculator.py # 计算奖励
│   └── aoi_calculator.py    # 计算和更新AoI
│
├── requirements.txt
├── README.md
└── .gitignore