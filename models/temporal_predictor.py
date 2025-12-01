"""
时间预测器网络
估计从当前状态到目标状态的期望时间步数
基于论文中的时间预测器：用于动态加权队列的优先级计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class TemporalPredictorConfig:
    """时间预测器配置"""
    # 输入维度
    state_dim: int  # 状态维度
    
    # 网络结构
    hidden_dims: List[int] = None  # 隐藏层维度
    activation: str = "relu"  # 激活函数
    
    # 输出
    output_dim: int = 1  # 输出维度（预测的时间步数）
    
    # 正则化
    dropout_rate: float = 0.2  # Dropout率
    use_batch_norm: bool = True  # 是否使用批归一化
    
    # 特殊模块
    use_attention: bool = True  # 是否使用注意力机制
    attention_heads: int = 4  # 注意力头数
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class TemporalPredictorNetwork(nn.Module):
    """时间预测器神经网络"""
    
    def __init__(self, config: TemporalPredictorConfig):
        """
        初始化时间预测器网络
        
        Args:
            config: 网络配置
        """
        super().__init__()
        self.config = config
        
        # 状态编码器（处理当前状态）
        self.state_encoder = self._create_encoder(config.state_dim, config.hidden_dims[0])
        
        # 目标编码器（处理目标状态）
        self.goal_encoder = self._create_encoder(config.state_dim, config.hidden_dims[0])
        
        # 注意力机制（比较当前状态和目标状态）
        if config.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dims[0],
                num_heads=config.attention_heads,
                batch_first=True,
                dropout=config.dropout_rate
            )
            
            self.self_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dims[0],
                num_heads=config.attention_heads,
                batch_first=True,
                dropout=config.dropout_rate
            )
            
        # 特征融合网络
        fusion_layers = []
        current_dim = config.hidden_dims[0] * 2  # 拼接状态和目标特征
        
        for hidden_dim in config.hidden_dims[1:]:
            fusion_layers.append(nn.Linear(current_dim, hidden_dim))
            
            if config.use_batch_norm:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim))
                
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
        
        # 输出层
        self.output_layer = nn.Linear(current_dim, config.output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
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
                # Kaiming初始化（适合ReLU）
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # 注意力权重初始化
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0.0)
                    
    def forward(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播：预测从当前状态到目标状态的时间步数
        
        Args:
            current_state: 当前状态 [batch_size, state_dim]
            goal_state: 目标状态 [batch_size, state_dim]
            
        Returns:
            预测的时间步数 [batch_size, 1]
        """
        batch_size = current_state.size(0)
        
        # 编码当前状态和目标状态
        current_features = self.state_encoder(current_state)  # [batch, features]
        goal_features = self.goal_encoder(goal_state)  # [batch, features]
        
        if self.config.use_attention:
            # 重塑为序列格式 [batch, seq_len=1, features]
            current_features_seq = current_features.unsqueeze(1)
            goal_features_seq = goal_features.unsqueeze(1)
            
            # 交叉注意力：当前状态关注目标状态
            attended_current, _ = self.cross_attention(
                query=current_features_seq,
                key=goal_features_seq,
                value=goal_features_seq
            )
            
            # 自注意力：增强特征表示
            enhanced_current, _ = self.self_attention(
                query=attended_current,
                key=attended_current,
                value=attended_current
            )
            
            # 更新特征
            current_features = enhanced_current.squeeze(1)
            
        # 拼接特征
        combined_features = torch.cat([current_features, goal_features], dim=-1)
        
        # 特征融合
        fused_features = self.fusion_network(combined_features)
        
        # 输出预测
        time_prediction = self.output_layer(fused_features)
        
        # 使用ReLU确保非负预测
        time_prediction = F.relu(time_prediction) + 1e-6  # 避免零
        
        return time_prediction
        
    def predict_time(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> float:
        """
        预测时间步数（推理模式）
        
        Args:
            current_state: 当前状态
            goal_state: 目标状态
            
        Returns:
            预测的时间步数
        """
        self.eval()
        with torch.no_grad():
            prediction = self.forward(current_state, goal_state)
        return prediction.item()
        
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    loss_type: str = "huber") -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            predictions: 预测值 [batch_size, 1]
            targets: 真实值 [batch_size, 1]
            loss_type: 损失类型 ("huber", "mse", "mae")
            
        Returns:
            损失值
        """
        if loss_type == "huber":
            # Huber损失（对异常值更鲁棒）
            loss = F.smooth_l1_loss(predictions, targets, reduction='mean')
        elif loss_type == "mse":
            # 均方误差
            loss = F.mse_loss(predictions, targets, reduction='mean')
        elif loss_type == "mae":
            # 平均绝对误差
            loss = F.l1_loss(predictions, targets, reduction='mean')
        elif loss_type == "log_cosh":
            # Log-Cosh损失（平滑的MAE）
            loss = torch.log(torch.cosh(predictions - targets)).mean()
        else:
            raise ValueError(f"未知损失类型: {loss_type}")
            
        return loss


class MultiScaleTemporalPredictor(TemporalPredictorNetwork):
    """多尺度时间预测器 - 考虑不同时间尺度"""
    
    def __init__(self, config: TemporalPredictorConfig, num_scales: int = 3):
        """
        初始化多尺度时间预测器
        
        Args:
            config: 网络配置
            num_scales: 尺度数量
        """
        super().__init__(config)
        self.num_scales = num_scales
        
        # 多尺度特征提取
        self.scale_encoders = nn.ModuleList([
            self._create_scale_encoder(config.state_dim, config.hidden_dims[0] // num_scales)
            for _ in range(num_scales)
        ])
        
        # 尺度注意力
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[0] // num_scales,
            num_heads=2,
            batch_first=True
        )
        
        # 多尺度输出层
        self.scale_outputs = nn.ModuleList([
            nn.Linear(config.hidden_dims[-1], 1)
            for _ in range(num_scales)
        ])
        
        # 最终融合层
        self.final_fusion = nn.Linear(num_scales, 1)
        
    def _create_scale_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建尺度编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU()
        )
        
    def forward(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播（多尺度版本）
        
        Args:
            current_state: 当前状态
            goal_state: 目标状态
            
        Returns:
            预测的时间步数
        """
        batch_size = current_state.size(0)
        
        # 多尺度特征提取
        scale_features_list = []
        
        for i in range(self.num_scales):
            # 应用不同尺度的编码器
            scale_factor = 2 ** i  # 1x, 2x, 4x等
            scaled_current = current_state / scale_factor
            scaled_goal = goal_state / scale_factor
            
            # 编码
            current_features = self.scale_encoders[i](scaled_current)
            goal_features = self.scale_encoders[i](scaled_goal)
            
            # 尺度注意力
            current_seq = current_features.unsqueeze(1)
            goal_seq = goal_features.unsqueeze(1)
            
            attended_current, _ = self.scale_attention(
                query=current_seq,
                key=goal_seq,
                value=goal_seq
            )
            
            scale_features = attended_current.squeeze(1)
            scale_features_list.append(scale_features)
            
        # 拼接所有尺度特征
        combined_scale_features = torch.cat(scale_features_list, dim=-1)
        
        # 通过融合网络
        fused_features = self.fusion_network(combined_scale_features)
        
        # 多尺度预测
        scale_predictions = []
        for i in range(self.num_scales):
            scale_pred = self.scale_outputs[i](fused_features)
            scale_predictions.append(scale_pred)
            
        # 堆叠预测
        predictions_stack = torch.stack(scale_predictions, dim=-1)  # [batch, 1, num_scales]
        predictions_stack = predictions_stack.squeeze(1)  # [batch, num_scales]
        
        # 最终融合
        final_prediction = self.final_fusion(predictions_stack)
        final_prediction = F.relu(final_prediction) + 1e-6
        
        return final_prediction


class RecurrentTemporalPredictor(TemporalPredictorNetwork):
    """循环时间预测器 - 考虑时间序列"""
    
    def __init__(self, config: TemporalPredictorConfig, sequence_length: int = 10):
        """
        初始化循环时间预测器
        
        Args:
            config: 网络配置
            sequence_length: 序列长度
        """
        super().__init__(config)
        self.sequence_length = sequence_length
        
        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            input_size=config.state_dim,
            hidden_size=config.hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate if config.dropout_rate > 0 else 0.0,
            bidirectional=True
        )
        
        # LSTM隐藏层大小（双向）
        lstm_output_dim = config.hidden_dims[0] * 2
        
        # 调整融合网络输入维度
        self.fusion_network[0] = nn.Linear(lstm_output_dim + config.hidden_dims[0], config.hidden_dims[1])
        
    def forward(self, state_sequence: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播（序列版本）
        
        Args:
            state_sequence: 状态序列 [batch_size, seq_len, state_dim]
            goal_state: 目标状态 [batch_size, state_dim]
            
        Returns:
            预测的时间步数
        """
        batch_size = state_sequence.size(0)
        
        # LSTM编码状态序列
        lstm_output, (h_n, c_n) = self.lstm_encoder(state_sequence)
        
        # 使用最后一个时间步的隐藏状态
        sequence_features = lstm_output[:, -1, :]  # [batch, features]
        
        # 编码目标状态
        goal_features = self.goal_encoder(goal_state)
        
        # 拼接特征
        combined_features = torch.cat([sequence_features, goal_features], dim=-1)
        
        # 特征融合
        fused_features = self.fusion_network(combined_features)
        
        # 输出预测
        time_prediction = self.output_layer(fused_features)
        time_prediction = F.relu(time_prediction) + 1e-6
        
        return time_prediction


# 工具函数
def create_temporal_predictor(state_dim: int, model_type: str = "standard") -> TemporalPredictorNetwork:
    """
    创建时间预测器工厂函数
    
    Args:
        state_dim: 状态维度
        model_type: 模型类型 ("standard", "multiscale", "recurrent")
        
    Returns:
        时间预测器网络
    """
    config = TemporalPredictorConfig(
        state_dim=state_dim,
        hidden_dims=[256, 128, 64],
        activation="relu",
        output_dim=1,
        use_attention=True
    )
    
    if model_type == "standard":
        return TemporalPredictorNetwork(config)
    elif model_type == "multiscale":
        return MultiScaleTemporalPredictor(config, num_scales=3)
    elif model_type == "recurrent":
        return RecurrentTemporalPredictor(config, sequence_length=10)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def test_temporal_predictor():
    """测试时间预测器"""
    print("测试时间预测器网络...")
    
    # 创建标准预测器
    state_dim = 50
    
    predictor = create_temporal_predictor(state_dim, "standard")
    print(f"创建标准预测器: {predictor}")
    
    # 测试前向传播
    batch_size = 4
    current_state = torch.randn(batch_size, state_dim)
    goal_state = torch.randn(batch_size, state_dim)
    
    prediction = predictor(current_state, goal_state)
    print(f"当前状态形状: {current_state.shape}")
    print(f"目标状态形状: {goal_state.shape}")
    print(f"预测形状: {prediction.shape}")
    print(f"预测值: {prediction.squeeze().detach().numpy()}")
    
    # 测试损失计算
    targets = torch.randint(1, 100, (batch_size, 1)).float()
    loss = predictor.compute_loss(prediction, targets, "huber")
    print(f"损失值: {loss.item()}")
    
    # 测试推理模式
    single_prediction = predictor.predict_time(current_state[0:1], goal_state[0:1])
    print(f"单样本预测: {single_prediction}")
    
    print("\n测试多尺度预测器...")
    # 测试多尺度预测器
    multiscale_predictor = create_temporal_predictor(state_dim, "multiscale")
    multiscale_prediction = multiscale_predictor(current_state, goal_state)
    print(f"多尺度预测形状: {multiscale_prediction.shape}")
    
    print("\n测试循环预测器...")
    # 测试循环预测器
    recurrent_predictor = create_temporal_predictor(state_dim, "recurrent")
    sequence_length = 10
    state_sequence = torch.randn(batch_size, sequence_length, state_dim)
    
    recurrent_prediction = recurrent_predictor(state_sequence, goal_state)
    print(f"状态序列形状: {state_sequence.shape}")
    print(f"循环预测形状: {recurrent_prediction.shape}")
    
    print("\n时间预测器测试完成!")


if __name__ == "__main__":
    test_temporal_predictor()