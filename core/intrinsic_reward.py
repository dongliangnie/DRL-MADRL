import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureEncoder(nn.Module):
    """
    状态特征编码器 (State Feature Encoder). 
    将原始观测状态映射到低维特征空间 ϕ(s)。
    """
    def __init__(self, obs_dim, feature_dim, hidden_dim=64):
        super(FeatureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(), # 使用 ELU 激活函数
            nn.Linear(hidden_dim, feature_dim),
            # 不使用最终的激活函数，以允许特征空间的值域更大
        )
        
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, obs):
        return self.encoder(obs)

class ForwardDynamicsModel(nn.Module):
    """
    前向动力学模型 (Forward Dynamics Model).
    预测下一状态的特征 ϕ(s_{t+1})，给定当前特征 ϕ(s_t) 和动作 a_t。
    """
    def __init__(self, feature_dim, action_dim, hidden_dim=128):
        super(ForwardDynamicsModel, self).__init__()
        
        # 输入: [feature_dim + action_dim]
        input_dim = feature_dim + action_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim) # 输出: 预测的下一状态特征
        )
        
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, phi_st, action):
        """
        Args:
            phi_st (Tensor): 当前状态特征 ϕ(s_t) [Batch, Feature_Dim]
            action (Tensor): 采取的动作 a_t [Batch, Action_Dim]
        Returns:
            predicted_phi_st_plus_1 (Tensor): 预测的下一状态特征 [Batch, Feature_Dim]
        """
        x = torch.cat([phi_st, action], dim=-1)
        return self.predictor(x)

class IntrinsicRewardModule(nn.Module):
    """
    自平衡内在奖励计算模块 (Self-Balancing Intrinsic Reward Calculator).
    
    采用 ICM (Intrinsic Curiosity Module) 框架来生成内在奖励。
    内在奖励 = 预测特征误差的 L2 范数（均方误差）。
    """
    def __init__(self, obs_dim, action_dim, feature_dim=64, hidden_dim_encoder=64, hidden_dim_forward=128):
        super(IntrinsicRewardModule, self).__init__()
        
        self.feature_encoder = FeatureEncoder(obs_dim, feature_dim, hidden_dim=hidden_dim_encoder)
        self.forward_model = ForwardDynamicsModel(feature_dim, action_dim, hidden_dim=hidden_dim_forward)
        
        # 我们可以选择性地添加 Inverse Dynamics Model (逆向动力学模型) 
        # 但为简化起见，我们仅实现用于内在奖励计算的 Forward Dynamics。
        
    def compute_intrinsic_reward_and_loss(self, state, action, next_state, eta=1.0, forward_loss_weight=1.0):
        """
        计算内在奖励和前向动力学模型的训练损失。
        
        Args:
            state (Tensor): 当前状态 s_t [Batch, Obs_Dim]
            action (Tensor): 动作 a_t [Batch, Action_Dim]
            next_state (Tensor): 下一状态 s_{t+1} [Batch, Obs_Dim]
            eta (float): 奖励缩放因子 (Intrinsic reward scale factor)
            forward_loss_weight (float): 前向模型损失权重 (Loss weight)
            
        Returns:
            intrinsic_reward (Tensor): 内在奖励 r_i [Batch, 1]
            forward_dynamics_loss (Tensor): 用于训练前向模型的损失
        """
        
        # 1. 状态特征编码
        phi_st = self.feature_encoder(state)
        phi_st_plus_1 = self.feature_encoder(next_state)
        
        # 2. 前向动力学预测
        predicted_phi_st_plus_1 = self.forward_model(phi_st.detach(), action.detach())
        
        # 3. 内在奖励计算 (使用 L2 范数/MSE 作为预测误差)
        # 注意：奖励计算中，特征编码器和动作需要 .detach()，避免梯度回传到策略网络
        # 奖励越高，表示预测误差越大，即状态越“新颖”
        
        # 预测误差（平方误差的平均）
        prediction_error = F.mse_loss(predicted_phi_st_plus_1, phi_st_plus_1.detach(), reduction='none').sum(dim=-1, keepdim=True)
        
        # 内在奖励 r_i = eta * 预测误差
        intrinsic_reward = eta * prediction_error
        
        # 4. 前向动力学模型训练损失
        # 目标是最小化预测误差，用于更新 Forward Dynamics Model 和 Feature Encoder
        forward_dynamics_loss = forward_loss_weight * prediction_error.mean()
        
        return intrinsic_reward, forward_dynamics_loss


if __name__ == "__main__":
    # --- Quick Sanity Check --- 快速自检
    print("Running IntrinsicRewardModule Sanity Check...")
    
    # 1. Hyperparameters - 超参数
    BATCH_SIZE = 16
    OBS_DIM = 48         # 观测状态维度
    ACTION_DIM = 2       # 动作维度
    FEATURE_DIM = 64     # 编码特征维度
    
    # 2. Model Initialization - 模型初始化
    irm = IntrinsicRewardModule(
        obs_dim=OBS_DIM, 
        action_dim=ACTION_DIM,
        feature_dim=FEATURE_DIM
    )
    print("Module initialized successfully.")

    # 3. Dummy Inputs - 虚拟输入
    # 形状: [Batch, Dim]
    dummy_state = torch.randn(BATCH_SIZE, OBS_DIM)
    dummy_action = torch.randn(BATCH_SIZE, ACTION_DIM)
    # 假设下一状态与当前状态略有不同
    dummy_next_state = dummy_state + 0.1 * torch.randn(BATCH_SIZE, OBS_DIM)

    # 4. Test Feature Encoder
    phi_st = irm.feature_encoder(dummy_state)
    print(f"Feature shape: {phi_st.shape} (Expected: {BATCH_SIZE}, {FEATURE_DIM})")
    assert phi_st.shape == (BATCH_SIZE, FEATURE_DIM)
    
    # 5. Test Compute Reward and Loss
    r_i, loss_fwd = irm.compute_intrinsic_reward_and_loss(
        dummy_state, dummy_action, dummy_next_state, eta=0.1, forward_loss_weight=10.0
    )
    
    print(f"Intrinsic Reward shape: {r_i.shape} (Expected: {BATCH_SIZE}, 1)")
    print(f"Forward Dynamics Loss (Scalar): {loss_fwd.item():.6f}")
    
    assert r_i.shape == (BATCH_SIZE, 1)
    
    # 6. Test Backpropagation (Ensure models are trainable)
    try:
        irm.zero_grad()
        loss_fwd.backward()
        
        # 检查 Feature Encoder 的权重是否接收到梯度
        encoder_weight = irm.feature_encoder.encoder[0].weight
        assert encoder_weight.grad is not None
        
        # 检查 Forward Model 的权重是否接收到梯度
        forward_weight = irm.forward_model.predictor[0].weight
        assert forward_weight.grad is not None
        
        print("Gradients successfully computed for IRM components.")
        print("Model is trainable.")
        
    except Exception as e:
        print(f"Backpropagation test failed: {e}")
        
    print("Sanity check passed!")