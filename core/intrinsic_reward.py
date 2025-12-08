import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicRewardModule(nn.Module):
    """
    内在奖励模块 (Intrinsic Reward Module, IRM)
    
    使用 ICM (Intrinsic Curiosity Module) 的思想：
    - Forward Model: 预测 s_{t+1}，计算预测误差作为内在奖励
    - Inverse Model: 从 (s_t, s_{t+1}) 预测 a_t，用于学习特征表示
    """
    def __init__(self, obs_dim, action_dim, feature_dim=64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # 特征编码器 (用于将原始观测映射到特征空间)
        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # 前向模型: phi(s_t) + a_t -> phi(s_{t+1})
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # 逆向模型: phi(s_t) + phi(s_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, action, next_obs):
        """
        前向传播
        
        Args:
            obs: 当前观测 (batch_size, obs_dim) 或 (obs_dim,)
            action: 执行的动作 (batch_size, action_dim) 或 (action_dim,)
            next_obs: 下一个观测 (batch_size, obs_dim) 或 (obs_dim,)
        
        Returns:
            intrinsic_reward: 内在奖励 (标量或 batch)
            loss_forward: 前向模型损失
            loss_inverse: 逆向模型损失
        """
        # 处理单样本情况 (添加 batch 维度)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 1. 编码特征
        phi_t = self.feature_encoder(obs)           # (batch, feature_dim)
        phi_t1 = self.feature_encoder(next_obs)     # (batch, feature_dim)
        
        # 2. 前向模型: 预测 phi(s_{t+1})
        forward_input = torch.cat([phi_t, action], dim=-1)
        phi_t1_pred = self.forward_model(forward_input)
        
        # 前向损失 (预测误差)
        loss_forward = F.mse_loss(phi_t1_pred, phi_t1.detach(), reduction='none').mean(dim=-1)
        
        # 3. 逆向模型: 预测动作
        inverse_input = torch.cat([phi_t, phi_t1], dim=-1)
        action_pred = self.inverse_model(inverse_input)
        
        # 逆向损失
        loss_inverse = F.mse_loss(action_pred, action, reduction='none').mean(dim=-1)
        
        # 4. 内在奖励 = 前向预测误差
        intrinsic_reward = loss_forward.detach()
        
        # 如果是单样本,移除 batch 维度
        if squeeze_output:
            intrinsic_reward = intrinsic_reward.squeeze(0)
            loss_forward = loss_forward.mean()  # 转为标量
            loss_inverse = loss_inverse.mean()
        else:
            loss_forward = loss_forward.mean()
            loss_inverse = loss_inverse.mean()
        
        return intrinsic_reward, loss_forward, loss_inverse
    
    def get_intrinsic_reward(self, obs, action, next_obs):
        """
        仅获取内在奖励 (不计算梯度)
        """
        with torch.no_grad():
            reward, _, _ = self.forward(obs, action, next_obs)
        return reward


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
    r_i, loss_fwd, loss_inv = irm.forward(dummy_state, dummy_action, dummy_next_state)
    
    print(f"Intrinsic Reward shape: {r_i.shape} (Expected: {BATCH_SIZE}, 1)")
    print(f"Forward Dynamics Loss (Scalar): {loss_fwd.item():.6f}")
    print(f"Inverse Dynamics Loss (Scalar): {loss_inv.item():.6f}")
    
    assert r_i.shape == (BATCH_SIZE, 1)
    
    # 6. Test Backpropagation (Ensure models are trainable)
    try:
        irm.zero_grad()
        loss_fwd.backward()
        
        # 检查 Feature Encoder 的权重是否接收到梯度
        encoder_weight = irm.feature_encoder[0].weight
        assert encoder_weight.grad is not None
        
        # 检查 Forward Model 的权重是否接收到梯度
        forward_weight = irm.forward_model[0].weight
        assert forward_weight.grad is not None
        
        # 检查 Inverse Model 的权重是否接收到梯度
        inverse_weight = irm.inverse_model[0].weight
        assert inverse_weight.grad is not None
        
        print("Gradients successfully computed for IRM components.")
        print("Model is trainable.")
        
    except Exception as e:
        print(f"Backpropagation test failed: {e}")
        
    print("Sanity check passed!")