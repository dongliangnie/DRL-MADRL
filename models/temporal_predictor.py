import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalPredictor(nn.Module):
    """
    Temporal Predictor Network (时间预测器网络).
    
    This module predicts future states (e.g., PoI AoI, UAV energy levels) 
    based on a sequence of past observations using a Recurrent Neural Network (GRU).
    
    Structure:
    - Input: Sequence of historical state observations [Batch, Seq_Len, Input_Dim]
    - Architecture:
        - GRU: Extracts temporal features from the sequence.
        - MLP: Maps the final hidden state of the GRU to the desired prediction output.
        
    Note: The initial hidden state of the GRU is typically zero.
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_size=128, 
                 num_layers=2):
        super(TemporalPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # -----------------------------------------------------------
        # 1. GRU Layer - 提取时间依赖性
        # batch_first=True means input shape is (batch, seq, feature)
        # -----------------------------------------------------------
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # -----------------------------------------------------------
        # 2. Prediction MLP - 将 GRU 的最终隐藏状态映射到输出
        # -----------------------------------------------------------
        self.predictor_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim) # 预测输出维度
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Standard initialization for stability.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # RNN weights often use orthogonal initialization
                    for w in param.chunk(self.gru.num_layers * 3, dim=0):
                        nn.init.orthogonal_(w)
                elif 'mlp' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, history_sequence):
        """
        Performs the forward pass to predict the next state.
        
        Args:
            history_sequence (Tensor): 
                历史状态序列，形状为 [Batch_Size, Sequence_Length, Input_Dim].
                
        Returns:
            prediction (Tensor): 
                预测输出，形状为 [Batch_Size, Output_Dim].
        """
        batch_size = history_sequence.size(0)
        
        # 1. GRU Forward Pass
        # hidden_state shape: [num_layers * num_directions, batch, hidden_size]
        # output shape: [batch, seq_len, hidden_size * num_directions] (ignored here)
        _, final_hidden_state = self.gru(history_sequence)
        
        # 2. Extract the hidden state from the last layer 
        # For simplicity, we use the hidden state of the LAST layer for prediction.
        # final_hidden_state[-1, :, :] has shape [batch, hidden_size]
        last_layer_hidden = final_hidden_state[-1, :, :]
        
        # 3. MLP Prediction
        prediction = self.predictor_mlp(last_layer_hidden)
        
        return prediction

if __name__ == "__main__":
    # --- Quick Sanity Check --- 快速自检
    print("Running TemporalPredictor Sanity Check...")
    
    # 1. Hyperparameters - 超参数
    BATCH_SIZE = 8
    SEQ_LEN = 5          # 观察的历史步长
    INPUT_DIM = 64       # 历史观测的特征维度 (例如, 组合了PoI和UAV特征)
    OUTPUT_DIM = 2       # 预测输出维度 (例如, 预测下一个AoI和能源变化)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    
    # 2. Model Initialization - 模型初始化
    predictor = TemporalPredictor(
        input_dim=INPUT_DIM, 
        output_dim=OUTPUT_DIM, 
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )
    print("Model initialized successfully.")

    # 3. Dummy Inputs - 虚拟输入
    # 形状: [Batch, Seq_Len, Input_Dim]
    dummy_sequence = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # 4. Test Forward Pass
    prediction = predictor(dummy_sequence)
    print(f"Input sequence shape: {dummy_sequence.shape}")
    print(f"Prediction shape: {prediction.shape} (Expected: {BATCH_SIZE}, {OUTPUT_DIM})")
    
    assert prediction.shape == (BATCH_SIZE, OUTPUT_DIM)
    
    # 5. Test Backpropagation (Ensure it is trainable)
    try:
        # 虚拟损失（例如，与随机目标的 MSE 损失）
        target = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        loss = F.mse_loss(prediction, target)
        
        predictor.zero_grad()
        loss.backward()
        
        # 检查 GRU 权重是否接收到梯度
        gru_weight_name = 'gru.weight_ih_l0'
        grad = predictor.state_dict()[gru_weight_name].grad
        assert grad is not None
        
        print(f"Loss: {loss.item():.4f}. Gradients successfully computed for {gru_weight_name}.")
        print("Model is trainable.")
        
    except Exception as e:
        print(f"Backpropagation test failed: {e}")
        
    print("Sanity check passed!")