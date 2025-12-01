"""
高层分配器网络
负责动态分配紧急任务给无人机
基于论文中的高层MDP：负责UAV目标分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class HighLevelAllocatorConfig:
    """高层分配器配置"""
    # 输入维度
    state_dim: int  # 全局状态维度
    
    # 网络结构
    hidden_dims: List[int] = None  # 隐藏层维度
    activation: str = "relu"  # 激活函数
    
    # 输出
    num_uavs: int = 4  # UAV数量（动作空间）
    
    # 正则化
    dropout_rate: float = 0.1  # Dropout率
    use_batch_norm: bool = False  # 是否使用批归一化
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class HighLevelAllocatorNetwork(nn.Module):
    """高层分配器神经网络"""
    
    def __init__(self, config: HighLevelAllocatorConfig):
        """
        初始化高层分配器网络
        
        Args:
            config: 网络配置
        """
        super().__init__()
        self.config = config
        
        # 创建网络层
        layers = []
        input_dim = config.state_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            # 线性层
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # 批归一化
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            # 激活函数
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"未知激活函数: {config.activation}")
                
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
                
            input_dim = hidden_dim
            
        # 输出层
        self.feature_extractor = nn.Sequential(*layers)
        
        # 策略头（输出动作概率）
        self.policy_head = nn.Linear(input_dim, config.num_uavs)
        
        # 价值头（输出状态价值）
        self.value_head = nn.Linear(input_dim, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 全局状态张量 [batch_size, state_dim]
            
        Returns:
            action_logits: 动作对数概率 [batch_size, num_uavs]
            value: 状态价值 [batch_size, 1]
        """
        # 提取特征
        features = self.feature_extractor(state)
        
        # 计算动作对数概率
        action_logits = self.policy_head(features)
        
        # 计算状态价值
        value = self.value_head(features)
        
        return action_logits, value
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作（UAV索引）
            action_prob: 动作概率
            value: 状态价值
        """
        # 前向传播
        action_logits, value = self.forward(state)
        
        # 计算动作概率
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            # 确定性选择：选择概率最高的动作
            action = torch.argmax(action_probs, dim=-1)
        else:
            # 随机采样
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            
        return action.item(), action_probs, value
        
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态和动作
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size]
            
        Returns:
            log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
            value: 状态价值 [batch_size]
        """
        # 前向传播
        action_logits, value = self.forward(state)
        
        # 创建分布
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # 计算对数概率
        log_probs = action_dist.log_prob(action)
        
        # 计算熵
        entropy = action_dist.entropy()
        
        return log_probs, entropy, value.squeeze(-1)


# class MultiHeadHighLevelAllocator(HighLevelAllocatorNetwork):
#     """多头高层分配器 - 支持多任务分配"""
    
#     def __init__(self, config: HighLevelAllocatorConfig, num_heads: int = 2):
#         """
#         初始化多头分配器
        
#         Args:
#             config: 网络配置
#             num_heads: 头数（例如，同时分配多个紧急任务）
#         """
#         super().__init__(config)
#         self.num_heads = num_heads
        
#         # 多头策略头
#         self.policy_heads = nn.ModuleList([
#             nn.Linear(config.hidden_dims[-1], config.num_uavs)
#             for _ in range(num_heads)
#         ])
        
#         # 多头注意力机制
#         self.attention = nn.MultiheadAttention(
#             embed_dim=config.hidden_dims[-1],
#             num_heads=4
#             # batch_first=True
#         )
        
#     def forward(self, state: torch.Tensor, task_features: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
#         """
#         前向传播（多头版本）
        
#         Args:
#             state: 全局状态张量 [batch_size, state_dim]
#             task_features: 任务特征张量 [batch_size, num_tasks, feature_dim]
            
#         Returns:
#             action_logits_list: 每个头的动作对数概率列表
#             value: 状态价值
#         """
#         # 提取特征
#         features = self.feature_extractor(state)
        
#         if task_features is not None and self.num_heads > 1:
#             # 多头注意力：考虑任务特征
#             batch_size = features.size(0)
#             features_expanded = features.unsqueeze(1).expand(-1, task_features.size(1), -1)
            
#             # 注意力机制
#             attended_features, _ = self.attention(
#                 query=task_features,
#                 key=features_expanded,
#                 value=features_expanded
#             )
            
#             # 聚合任务特征
#             task_aware_features = attended_features.mean(dim=1)
#             features = features + task_aware_features  # 残差连接
            
#         # 计算每个头的动作对数概率
#         action_logits_list = []
#         for i in range(self.num_heads):
#             action_logits = self.policy_heads[i](features)
#             action_logits_list.append(action_logits)
            
#         # 计算状态价值
#         value = self.value_head(features)
        
#         return action_logits_list, value
        
#     def get_multi_actions(self, state: torch.Tensor, task_features: Optional[torch.Tensor] = None) -> List[Tuple[int, torch.Tensor]]:
#         """
#         获取多个动作分配
        
#         Args:
#             state: 状态张量
#             task_features: 任务特征
            
#         Returns:
#             动作列表 [(action1, prob1), ...]
#         """
#         action_logits_list, value = self.forward(state, task_features)
        
#         actions = []
#         for action_logits in action_logits_list:
#             action_probs = F.softmax(action_logits, dim=-1)
#             action = torch.argmax(action_probs, dim=-1)
#             actions.append((action.item(), action_probs))
            
#         return actions
class MultiHeadHighLevelAllocator(HighLevelAllocatorNetwork):
    """多头高层分配器 - 支持多任务分配"""

    def __init__(self, config: HighLevelAllocatorConfig, num_heads: int = 2):
        super().__init__(config)
        self.num_heads = num_heads

        hidden_dim = config.hidden_dims[-1]

        # 多头策略：每个头输出一个 UAV 分配分布
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, config.num_uavs)
            for _ in range(num_heads)
        ])

        # 注意力 embed_dim = hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            # batch_first=True
        )

        # 将 task_features 映射到 hidden_dim（避免维度不一致）
        self.task_feature_proj = nn.Linear(config.task_feature_dim, hidden_dim)

        # 将 features 扩展后映射到 hidden_dim（必要时）
        self.state_feature_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor, task_features: Optional[torch.Tensor] = None):
        """
        前向传播

        Args:
            state: [batch, state_dim]
            task_features: [batch, num_tasks, task_feature_dim]

        Returns:
            action_logits_list: 多头动作 logits
            value: 状态价值
        """
        # 提取全局状态特征
        features = self.feature_extractor(state)                    # [B, hidden]

        # 如果有任务特征，则使用注意力增强
        if task_features is not None and task_features.size(1) > 0:

            batch_size, num_tasks, _ = task_features.shape
            hidden_dim = features.size(-1)

            # ---- 1. 投影到 hidden_dim ----
            task_feats = self.task_feature_proj(task_features)      # [B, T, H]
            state_feats = self.state_feature_proj(features)          # [B, H]

            # ---- 2. 扩展状态特征到序列 ----
            state_feats_expanded = state_feats.unsqueeze(1).expand(-1, num_tasks, -1)  # [B, T, H]

            # ---- 3. 注意力：任务作为 Query，状态作为 Key/Value ----
            attended, _ = self.attention(
                query=task_feats,               # [B, T, H]
                key=state_feats_expanded,       # [B, T, H]
                value=state_feats_expanded      # [B, T, H]
            )

            # ---- 4. 聚合任务特征 ----
            task_aware_feature = attended.mean(dim=1)                # [B, H]

            # ---- 5. 残差增强 ----
            features = features + task_aware_feature

        # ---- 输出每个头的策略 ----
        action_logits_list = [ head(features) for head in self.policy_heads ]

        # ---- 状态价值 ----
        value = self.value_head(features)

        return action_logits_list, value

    def get_multi_actions(self, state: torch.Tensor, task_features: Optional[torch.Tensor] = None):
        """
        选择多个动作（每个 head 一个）
        """
        action_logits_list, value = self.forward(state, task_features)

        actions = []
        for logits in action_logits_list:
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
            actions.append((action.item(), probs))

        return actions


class AttentionBasedAllocator(HighLevelAllocatorNetwork):
    """基于注意力的高层分配器"""
    
    def __init__(self, config: HighLevelAllocatorConfig, attention_dim: int = 128):
        """
        初始化注意力分配器
        
        Args:
            config: 网络配置
            attention_dim: 注意力维度
        """
        super().__init__(config)
        
        # UAV特征编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(10, 64),  # UAV状态维度
            nn.ReLU(),
            nn.Linear(64, attention_dim)
        )
        
        # 任务特征编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(8, 64),  # 任务状态维度
            nn.ReLU(),
            nn.Linear(64, attention_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(attention_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, attention_dim)
        )
        
    def forward(self, uav_states: torch.Tensor, task_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（基于注意力）
        
        Args:
            uav_states: UAV状态 [batch_size, num_uavs, uav_state_dim]
            task_states: 任务状态 [batch_size, num_tasks, task_state_dim]
            
        Returns:
            action_logits: 动作对数概率 [batch_size, num_uavs * num_tasks]
            value: 状态价值 [batch_size, 1]
        """
        batch_size = uav_states.size(0)
        num_uavs = uav_states.size(1)
        num_tasks = task_states.size(1)
        
        # 编码UAV特征
        uav_features = self.uav_encoder(uav_states.view(batch_size * num_uavs, -1))
        uav_features = uav_features.view(batch_size, num_uavs, -1)
        
        # 编码任务特征
        task_features = self.task_encoder(task_states.view(batch_size * num_tasks, -1))
        task_features = task_features.view(batch_size, num_tasks, -1)
        
        # 注意力机制：任务关注UAV
        attended_features, attention_weights = self.attention(
            query=task_features,
            key=uav_features,
            value=uav_features
        )
        
        # 融合特征
        # 任务特征与注意力特征的融合
        fused_features = torch.cat([task_features, attended_features], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # 全局池化
        global_features = fused_features.mean(dim=1)
        
        # 通过基础网络
        action_logits, value = super().forward(global_features)
        
        return action_logits, value, attention_weights
        
    def visualize_attention(self, uav_states: torch.Tensor, task_states: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        可视化注意力权重
        
        Args:
            uav_states: UAV状态
            task_states: 任务状态
            
        Returns:
            注意力权重可视化数据
        """
        _, _, attention_weights = self.forward(uav_states, task_states)
        
        # 将注意力权重转换为numpy数组
        attention_np = attention_weights.detach().cpu().numpy()
        
        # 计算每个UAV对每个任务的重要性
        uav_task_importance = attention_np.mean(axis=1)  # 平均多头注意力
        
        return {
            'attention_weights': attention_np,
            'uav_task_importance': uav_task_importance,
            'num_uavs': uav_states.size(1),
            'num_tasks': task_states.size(1)
        }


# 工具函数
def create_high_level_allocator(state_dim: int, num_uavs: int, model_type: str = "standard") -> HighLevelAllocatorNetwork:
    """
    创建高层分配器工厂函数
    
    Args:
        state_dim: 状态维度
        num_uavs: UAV数量
        model_type: 模型类型 ("standard", "multihead", "attention")
        
    Returns:
        高层分配器网络
    """
    config = HighLevelAllocatorConfig(
        state_dim=state_dim,
        num_uavs=num_uavs,
        hidden_dims=[256, 128],
        activation="relu"
    )
    
    if model_type == "standard":
        return HighLevelAllocatorNetwork(config)
    elif model_type == "multihead":
        return MultiHeadHighLevelAllocator(config, num_heads=2)
    elif model_type == "attention":
        return AttentionBasedAllocator(config)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def test_high_level_allocator():
    """测试高层分配器"""
    print("测试高层分配器网络...")
    
    # 创建标准分配器
    state_dim = 100
    num_uavs = 4
    
    allocator = create_high_level_allocator(state_dim, num_uavs, "standard")
    print(f"创建标准分配器: {allocator}")
    
    # 测试前向传播
    batch_size = 2
    state = torch.randn(batch_size, state_dim)
    
    action_logits, value = allocator(state)
    print(f"输入形状: {state.shape}")
    print(f"动作对数概率形状: {action_logits.shape}")
    print(f"价值形状: {value.shape}")
    
    # 测试选择动作
    action, action_probs, value = allocator.get_action(state[0:1])
    print(f"选择动作: {action}")
    print(f"动作概率形状: {action_probs.shape}")
    
    # 测试评估动作
    actions = torch.randint(0, num_uavs, (batch_size,))
    log_probs, entropy, values = allocator.evaluate_actions(state, actions)
    print(f"对数概率形状: {log_probs.shape}")
    print(f"熵形状: {entropy.shape}")
    print(f"价值形状: {values.shape}")
    
    print("\n测试多头分配器...")
    # 测试多头分配器
    multihead_allocator = create_high_level_allocator(state_dim, num_uavs, "multihead")
    task_features = torch.randn(batch_size, 3, 32)  # 3个任务，每个32维特征
    
    action_logits_list, value = multihead_allocator(state, task_features)
    print(f"多头输出数量: {len(action_logits_list)}")
    
    # 测试基于注意力的分配器
    print("\n测试基于注意力的分配器...")
    attention_allocator = create_high_level_allocator(state_dim, num_uavs, "attention")
    
    uav_states = torch.randn(batch_size, num_uavs, 10)  # 每个UAV 10维状态
    task_states = torch.randn(batch_size, 3, 8)  # 每个任务 8维状态
    
    action_logits, value, attention_weights = attention_allocator(uav_states, task_states)
    print(f"注意力权重形状: {attention_weights.shape}")
    
    print("\n高层分配器测试完成!")


if __name__ == "__main__":
    test_high_level_allocator()