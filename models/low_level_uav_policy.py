"""
低层UAV策略网络
负责单个UAV的移动决策，基于局部观察和分配的目标
基于论文中的低层MDP：负责多任务导向的UAV执行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class LowLevelUAVPolicyConfig:
    """低层UAV策略配置"""
    # 输入维度
    observation_dim: int  # 观察维度
    goal_dim: int  # 目标维度
    
    # 网络结构
    hidden_dims: List[int] = None  # 隐藏层维度
    activation: str = "relu"  # 激活函数
    
    # 输出
    action_dim: int = 2  # 动作维度（速度向量）
    
    # 正则化
    dropout_rate: float = 0.1  # Dropout率
    use_batch_norm: bool = True  # 是否使用批归一化
    use_layer_norm: bool = False  # 是否使用层归一化
    
    # 特殊模块
    use_attention: bool = False  # 是否使用注意力机制
    use_lstm: bool = False  # 是否使用LSTM记忆
    lstm_hidden_size: int = 128  # LSTM隐藏层大小
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class LowLevelUAVPolicyNetwork(nn.Module):
    """低层UAV策略网络（Actor-Critic架构）"""
    
    def __init__(self, config: LowLevelUAVPolicyConfig):
        """
        初始化低层UAV策略网络
        
        Args:
            config: 网络配置
        """
        super().__init__()
        self.config = config
        
        # 观察编码器
        self.obs_encoder = self._create_encoder(config.observation_dim, config.hidden_dims[0])
        
        # 目标编码器
        self.goal_encoder = self._create_encoder(config.goal_dim, config.hidden_dims[0])
        
        # LSTM记忆（如果启用）
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_dims[0] * 2,
                hidden_size=config.lstm_hidden_size,
                batch_first=True
            )
            feature_dim = config.lstm_hidden_size
        else:
            self.lstm = None
            feature_dim = config.hidden_dims[0] * 2
            
        # 特征融合层
        fusion_layers = []
        current_dim = feature_dim
        
        for hidden_dim in config.hidden_dims[1:]:
            fusion_layers.append(nn.Linear(current_dim, hidden_dim))
            
            if config.use_batch_norm:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                fusion_layers.append(nn.LayerNorm(hidden_dim))
                
            # 激活函数
            if config.activation == "relu":
                fusion_layers.append(nn.ReLU())
            elif config.activation == "leaky_relu":
                fusion_layers.append(nn.LeakyReLU(0.01))
            elif config.activation == "tanh":
                fusion_layers.append(nn.Tanh())
            elif config.activation == "elu":
                fusion_layers.append(nn.ELU())
                
            if config.dropout_rate > 0:
                fusion_layers.append(nn.Dropout(config.dropout_rate))
                
            current_dim = hidden_dim
            
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # 注意力机制（如果启用）
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=current_dim,
                num_heads=4,
                batch_first=True
            )
            attention_dim = current_dim
        else:
            self.attention = None
            attention_dim = current_dim
            
        # Actor头（输出动作均值和标准差）
        self.actor_mean = nn.Linear(attention_dim, config.action_dim)
        self.actor_log_std = nn.Linear(attention_dim, config.action_dim)
        
        # Critic头（输出状态价值）
        self.critic = nn.Linear(attention_dim, 1)
        
        # 初始化权重
        self._initialize_weights()
        
        # 初始化LSTM状态
        self.lstm_hidden_state = None
        self.lstm_cell_state = None
        
    def _create_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 正交初始化（适合RNN和深度网络）
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                # LSTM权重初始化
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        # 设置遗忘门偏置为1（有助于梯度流动）
                        if len(param) > 3:
                            param.data[3:].fill_(1.0)
                            
    def reset_lstm_state(self, batch_size: int = 1):
        """重置LSTM状态"""
        if self.lstm:
            device = next(self.parameters()).device
            hidden_size = self.config.lstm_hidden_size
            
            self.lstm_hidden_state = torch.zeros(1, batch_size, hidden_size, device=device)
            self.lstm_cell_state = torch.zeros(1, batch_size, hidden_size, device=device)
            
    def forward(self, 
                observation: torch.Tensor, 
                goal: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            observation: 观察张量 [batch_size, obs_dim]
            goal: 目标张量 [batch_size, goal_dim]
            hidden_state: LSTM隐藏状态 (h, c)
            
        Returns:
            action_mean: 动作均值 [batch_size, action_dim]
            action_log_std: 动作对数标准差 [batch_size, action_dim]
            value: 状态价值 [batch_size, 1]
        """
        batch_size = observation.size(0)
        
        # 编码观察和目标
        obs_features = self.obs_encoder(observation)
        goal_features = self.goal_encoder(goal)
        
        # 拼接特征
        combined_features = torch.cat([obs_features, goal_features], dim=-1)
        
        # LSTM处理（如果有时间序列）
        if self.lstm:
            # 重塑为序列格式 [batch_size, seq_len=1, features]
            combined_features = combined_features.unsqueeze(1)
            
            if hidden_state is None:
                if self.lstm_hidden_state is None:
                    self.reset_lstm_state(batch_size)
                hidden_state = (self.lstm_hidden_state, self.lstm_cell_state)
                
            # LSTM前向传播
            lstm_out, (h_n, c_n) = self.lstm(combined_features, hidden_state)
            
            # 更新隐藏状态
            self.lstm_hidden_state = h_n.detach()
            self.lstm_cell_state = c_n.detach()
            
            features = lstm_out.squeeze(1)
        else:
            features = combined_features
            
        # 特征融合
        fused_features = self.fusion_network(features)
        
        # 注意力机制（如果启用）
        if self.attention:
            # 自注意力
            attended_features, _ = self.attention(
                fused_features.unsqueeze(1),
                fused_features.unsqueeze(1),
                fused_features.unsqueeze(1)
            )
            fused_features = attended_features.squeeze(1)
            
        # 计算动作分布参数
        action_mean = self.actor_mean(fused_features)
        action_log_std = self.actor_log_std(fused_features)
        
        # 限制标准差范围
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        # 计算状态价值
        value = self.critic(fused_features)
        
        return action_mean, action_log_std, value
        
    def get_action(self, 
                   observation: torch.Tensor, 
                   goal: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            observation: 观察张量
            goal: 目标张量
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作
            log_prob: 动作对数概率
            value: 状态价值
        """
        # 前向传播
        action_mean, action_log_std, value = self.forward(observation, goal)
        action_std = torch.exp(action_log_std)
        
        if deterministic:
            # 确定性选择：选择均值
            action = action_mean
            log_prob = None
        else:
            # 随机采样：从正态分布采样
            normal_dist = torch.distributions.Normal(action_mean, action_std)
            action = normal_dist.rsample()  # 重参数化技巧
            
            # 计算对数概率
            log_prob = normal_dist.log_prob(action).sum(dim=-1)
            
            # 确保动作在有效范围内（-1到1）
            action = torch.tanh(action)
            
            # 调整对数概率（由于tanh变换）
            if log_prob is not None:
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
                
        return action, log_prob, value.squeeze(-1)
        
    def evaluate_actions(self, 
                         observation: torch.Tensor, 
                         goal: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Args:
            observation: 观察张量 [batch_size, obs_dim]
            goal: 目标张量 [batch_size, goal_dim]
            actions: 动作张量 [batch_size, action_dim]
            
        Returns:
            log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
            value: 状态价值 [batch_size]
        """
        # 前向传播
        action_mean, action_log_std, value = self.forward(observation, goal)
        action_std = torch.exp(action_log_std)
        
        # 创建分布
        normal_dist = torch.distributions.Normal(action_mean, action_std)
        
        # 计算对数概率
        log_probs = normal_dist.log_prob(actions).sum(dim=-1)
        
        # 计算熵
        entropy = normal_dist.entropy().sum(dim=-1)
        
        return log_probs, entropy, value.squeeze(-1)


class MultiTaskUAVPolicy(LowLevelUAVPolicyNetwork):
    """多任务UAV策略网络 - 支持多个任务类型的处理"""
    
    def __init__(self, config: LowLevelUAVPolicyConfig, num_task_types: int = 2):
        """
        初始化多任务策略网络
        
        Args:
            config: 网络配置
            num_task_types: 任务类型数量
        """
        super().__init__(config)
        self.num_task_types = num_task_types
        
        # 任务类型编码器
        self.task_type_embedding = nn.Embedding(num_task_types, 16)
        
        # 多个任务目标编码器
        self.task_encoders = nn.ModuleList([
            self._create_encoder(config.goal_dim, config.hidden_dims[0])
            for _ in range(num_task_types)
        ])
        
        # 任务注意力机制
        self.task_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[0],
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, 
                observation: torch.Tensor, 
                goals: List[torch.Tensor],
                task_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（多任务版本）
        
        Args:
            observation: 观察张量 [batch_size, obs_dim]
            goals: 目标张量列表，每个 [batch_size, goal_dim]
            task_types: 任务类型张量 [batch_size]
            
        Returns:
            action_mean: 动作均值
            action_log_std: 动作对数标准差
            value: 状态价值
        """
        batch_size = observation.size(0)
        
        # 编码观察
        obs_features = self.obs_encoder(observation)
        
        # 编码任务类型
        task_type_emb = self.task_type_embedding(task_types)
        
        # 编码每个任务目标
        goal_features_list = []
        for i, goal in enumerate(goals):
            if i < self.num_task_types:
                goal_features = self.task_encoders[i](goal)
                goal_features_list.append(goal_features)
                
        # 注意力机制：聚合任务特征
        if len(goal_features_list) > 0:
            goal_features_stack = torch.stack(goal_features_list, dim=1)  # [batch, num_tasks, features]
            
            # 任务注意力
            attended_goals, _ = self.task_attention(
                query=goal_features_stack,
                key=goal_features_stack,
                value=goal_features_stack
            )
            
            # 聚合任务特征
            aggregated_goals = attended_goals.mean(dim=1)
        else:
            aggregated_goals = torch.zeros_like(obs_features)
            
        # 拼接所有特征
        combined_features = torch.cat([
            obs_features,
            aggregated_goals,
            task_type_emb
        ], dim=-1)
        
        # 通过融合网络
        fused_features = self.fusion_network(combined_features)
        
        # 计算动作分布参数
        action_mean = self.actor_mean(fused_features)
        action_log_std = self.actor_log_std(fused_features)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        # 计算状态价值
        value = self.critic(fused_features)
        
        return action_mean, action_log_std, value


class HierarchicalUAVPolicy(nn.Module):
    """分层UAV策略网络 - 高层目标选择和低层执行"""
    
    def __init__(self, 
                 high_level_config: LowLevelUAVPolicyConfig,
                 low_level_config: LowLevelUAVPolicyConfig):
        """
        初始化分层策略网络
        
        Args:
            high_level_config: 高层网络配置
            low_level_config: 低层网络配置
        """
        super().__init__()
        
        # 高层网络：选择目标
        self.high_level_policy = LowLevelUAVPolicyNetwork(high_level_config)
        
        # 低层网络：执行动作
        self.low_level_policy = LowLevelUAVPolicyNetwork(low_level_config)
        
        # 目标编码器
        self.goal_encoder = nn.Sequential(
            nn.Linear(high_level_config.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, low_level_config.goal_dim)
        )
        
    def forward(self, 
                high_level_obs: torch.Tensor,
                low_level_obs: torch.Tensor,
                high_level_goal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（分层版本）
        
        Args:
            high_level_obs: 高层观察 [batch_size, high_obs_dim]
            low_level_obs: 低层观察 [batch_size, low_obs_dim]
            high_level_goal: 高层目标（可选）
            
        Returns:
            high_action: 高层动作（目标选择）
            low_action: 低层动作（移动）
            high_value: 高层价值
            low_value: 低层价值
        """
        # 高层决策：选择目标
        if high_level_goal is None:
            # 高层网络选择目标
            high_action_mean, high_action_log_std, high_value = self.high_level_policy(
                high_level_obs, 
                torch.zeros_like(high_level_obs[:, :self.high_level_policy.config.goal_dim])
            )
            
            # 采样高层动作
            high_action_std = torch.exp(high_action_log_std)
            high_normal_dist = torch.distributions.Normal(high_action_mean, high_action_std)
            high_action = high_normal_dist.rsample()
        else:
            high_action = high_level_goal
            high_value = None
            
        # 编码高层动作为低层目标
        low_level_goal = self.goal_encoder(high_action)
        
        # 低层决策：执行动作
        low_action_mean, low_action_log_std, low_value = self.low_level_policy(
            low_level_obs,
            low_level_goal
        )
        
        # 采样低层动作
        low_action_std = torch.exp(low_action_log_std)
        low_normal_dist = torch.distributions.Normal(low_action_mean, low_action_std)
        low_action = low_normal_dist.rsample()
        low_action = torch.tanh(low_action)  # 限制在[-1, 1]
        
        return high_action, low_action, high_value, low_value


# 工具函数
def create_low_level_policy(obs_dim: int, goal_dim: int, model_type: str = "standard") -> LowLevelUAVPolicyNetwork:
    """
    创建低层UAV策略工厂函数
    
    Args:
        obs_dim: 观察维度
        goal_dim: 目标维度
        model_type: 模型类型 ("standard", "multitask", "hierarchical")
        
    Returns:
        低层UAV策略网络
    """
    config = LowLevelUAVPolicyConfig(
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        hidden_dims=[256, 256],
        activation="relu",
        action_dim=2,
        use_lstm=True
    )
    
    if model_type == "standard":
        return LowLevelUAVPolicyNetwork(config)
    elif model_type == "multitask":
        return MultiTaskUAVPolicy(config, num_task_types=2)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def test_low_level_policy():
    """测试低层UAV策略"""
    print("测试低层UAV策略网络...")
    
    # 创建标准策略
    obs_dim = 50
    goal_dim = 20
    
    policy = create_low_level_policy(obs_dim, goal_dim, "standard")
    print(f"创建标准策略: {policy}")
    
    # 测试前向传播
    batch_size = 4
    observation = torch.randn(batch_size, obs_dim)
    goal = torch.randn(batch_size, goal_dim)
    
    # 重置LSTM状态
    policy.reset_lstm_state(batch_size)
    
    action_mean, action_log_std, value = policy(observation, goal)
    print(f"观察形状: {observation.shape}")
    print(f"目标形状: {goal.shape}")
    print(f"动作均值形状: {action_mean.shape}")
    print(f"动作对数标准差形状: {action_log_std.shape}")
    print(f"价值形状: {value.shape}")
    
    # 测试选择动作
    action, log_prob, value = policy.get_action(observation[0:1], goal[0:1])
    print(f"动作形状: {action.shape}")
    print(f"对数概率: {log_prob}")
    
    # 测试评估动作
    actions = torch.randn(batch_size, 2)
    log_probs, entropy, values = policy.evaluate_actions(observation, goal, actions)
    print(f"对数概率形状: {log_probs.shape}")
    print(f"熵形状: {entropy.shape}")
    print(f"价值形状: {values.shape}")
    
    print("\n测试多任务策略...")
    # 测试多任务策略
    multitask_policy = create_low_level_policy(obs_dim, goal_dim, "multitask")
    
    goals = [torch.randn(batch_size, goal_dim), torch.randn(batch_size, goal_dim)]
    task_types = torch.randint(0, 2, (batch_size,))
    
    action_mean, action_log_std, value = multitask_policy(observation, goals, task_types)
    print(f"多任务动作均值形状: {action_mean.shape}")
    
    print("\n测试分层策略...")
    # 测试分层策略
    high_config = LowLevelUAVPolicyConfig(
        observation_dim=100,
        goal_dim=30,
        action_dim=10
    )
    
    low_config = LowLevelUAVPolicyConfig(
        observation_dim=obs_dim,
        goal_dim=20,
        action_dim=2
    )
    
    hierarchical_policy = HierarchicalUAVPolicy(high_config, low_config)
    
    high_obs = torch.randn(batch_size, 100)
    low_obs = torch.randn(batch_size, obs_dim)
    
    high_action, low_action, high_value, low_value = hierarchical_policy(high_obs, low_obs)
    print(f"高层动作形状: {high_action.shape}")
    print(f"低层动作形状: {low_action.shape}")
    
    print("\n低层UAV策略测试完成!")


if __name__ == "__main__":
    test_low_level_policy()