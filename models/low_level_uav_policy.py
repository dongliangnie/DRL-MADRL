import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class LowLevelUAVPolicy(nn.Module):
    """
    Low-level policy for Multi-Task-Oriented UAV executions (Section IV.A & IV.D).
    
    Structure:
    - Input: 
        1. Local Observation (Vector): UAV location, energy, relative positions.
        2. Local Map (Image/Grid): Local AoI heatmap within sensing range.
        3. Goal (Vector): Assigned emergency PoI features (location, remaining time).
    - Architecture:
        - CNN: Processes Local Map.
        - MLP: Processes Vector Obs + Goal.
        - Fusion: Concatenates features.
        - Actor Head: Outputs action distribution (Mean & Std) for velocity control.
        - Critic Head: Outputs Value function V(s).
    
    Paper Settings (Section V.B): "3-layer MLP with 128 hidden states".
    """
    def __init__(self, 
                 vector_obs_dim, 
                 goal_dim, 
                 action_dim, 
                 map_channels=1, 
                 map_size=20, 
                 hidden_dim=128):
        super(LowLevelUAVPolicy, self).__init__()
        
        self.hidden_dim = hidden_dim

        # -----------------------------------------------------------
        # 1. Map Feature Extractor (CNN)
        # Extracts spatial features from the local AoI heatmap.
        # -----------------------------------------------------------
        self.cnn = nn.Sequential(
            # Input: [batch, map_channels, map_size, map_size]
            nn.Conv2d(map_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample by 2
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample by 2
            
            nn.Flatten()
        )
        
        # Calculate CNN output dimension dynamically
        with torch.no_grad():
            dummy_map = torch.zeros(1, map_channels, map_size, map_size)
            cnn_out_dim = self.cnn(dummy_map).shape[1]
            
        # Project CNN features to hidden_dim
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 2. Vector & Goal Feature Extractor (MLP)
        # Processes UAV status and Goal information.
        # -----------------------------------------------------------
        # Input: Vector Obs + Goal Embedding
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 3. Actor-Critic Heads
        # Input: Fused features (CNN features + MLP features)
        # -----------------------------------------------------------
        fusion_dim = hidden_dim * 2 # Concatenated size

        # Actor Network (Policy)
        # 3-layer MLP as per paper
        self.actor_body = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log_std for continuous action space (PPO standard)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Network (Value)
        self.critic_body = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs scalar V(s)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for stable PPO training.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _extract_features(self, vector_obs, map_obs, goal):
        """
        Internal method to process inputs and fuse features.
        """
        # 1. Process Map
        # Ensure map_obs is [batch, channels, H, W]
        if map_obs.dim() == 3: 
            map_obs = map_obs.unsqueeze(1) # Add channel dim if missing
            
        map_feat = self.cnn(map_obs)
        map_feat = self.cnn_fc(map_feat)
        
        # 2. Process Vector + Goal
        vec_input = torch.cat([vector_obs, goal], dim=-1)
        vec_feat = self.vector_mlp(vec_input)
        
        # 3. Fuse (Concatenate)
        fused_feat = torch.cat([map_feat, vec_feat], dim=-1)
        return fused_feat

    def get_distribution(self, vector_obs, map_obs, goal):
        """
        Construct the action distribution (Gaussian).
        """
        features = self._extract_features(vector_obs, map_obs, goal)
        
        x = self.actor_body(features)
        mean = torch.tanh(self.action_mean(x)) # Bound mean to [-1, 1]
        
        # Calculate Std
        log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return Normal(mean, std)

    def act(self, vector_obs, map_obs, goal, deterministic=False):
        """
        Select action for interaction (Inference).
        Returns:
            action: Action to execute (e.g., normalized velocity)
            log_prob: Log probability of the action (for PPO buffer)
        """
        dist = self.get_distribution(vector_obs, map_obs, goal)
        
        if deterministic:
            action = dist.mean
            log_prob = torch.zeros(action.shape[0], 1).to(action.device) # Placeholder
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
        return action, log_prob

    def get_value(self, vector_obs, map_obs, goal):
        """
        Get Value V(s) for the current state (Critic).
        """
        features = self._extract_features(vector_obs, map_obs, goal)
        value = self.critic_body(features)
        return value
    def forward(self, vector_obs, map_obs, goal):
        """
        Forward method required by nn.Module.
        This makes the module callable: policy(vec, map, goal)
        It returns deterministic action (mean of the policy distribution).
        """
        action, _ = self.act(vector_obs, map_obs, goal, deterministic=True)
        return action

    def evaluate_actions(self, vector_obs, map_obs, goal, actions):
        """
        Evaluate actions for PPO Update.
        Returns:
            value: V(s)
            action_log_probs: Log prob of the actions taken
            dist_entropy: Entropy of the distribution (for exploration bonus)
        """
        features = self._extract_features(vector_obs, map_obs, goal)
        
        # 1. Critic Value
        value = self.critic_body(features)
        
        # 2. Actor Distribution
        x = self.actor_body(features)
        mean = torch.tanh(self.action_mean(x))
        log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # 3. Metrics
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, action_log_probs, dist_entropy

if __name__ == "__main__":
    # --- Quick Sanity Check ---
    print("Running LowLevelUAVPolicy Sanity Check...")
    
    # 1. Hyperparameters
    BATCH_SIZE = 4
    VEC_DIM = 10
    GOAL_DIM = 5
    ACT_DIM = 2
    MAP_SIZE = 20
    MAP_CH = 1
    
    # 2. Model Initialization
    policy = LowLevelUAVPolicy(
        vector_obs_dim=VEC_DIM, 
        goal_dim=GOAL_DIM, 
        action_dim=ACT_DIM, 
        map_size=MAP_SIZE
    )
    print("Model initialized successfully.")

    # 3. Dummy Inputs
    dummy_vec = torch.randn(BATCH_SIZE, VEC_DIM)
    dummy_map = torch.randn(BATCH_SIZE, MAP_CH, MAP_SIZE, MAP_SIZE)
    dummy_goal = torch.randn(BATCH_SIZE, GOAL_DIM)

    # 4. Test Inference (act)
    action, log_prob = policy.act(dummy_vec, dummy_map, dummy_goal)
    print(f"Action shape: {action.shape} (Expected: {BATCH_SIZE}, {ACT_DIM})")
    print(f"LogProb shape: {log_prob.shape} (Expected: {BATCH_SIZE}, 1)")

    # 5. Test Value (get_value)
    value = policy.get_value(dummy_vec, dummy_map, dummy_goal)
    print(f"Value shape: {value.shape} (Expected: {BATCH_SIZE}, 1)")
    
    # 6. Test Evaluation (evaluate_actions)
    v, lp, ent = policy.evaluate_actions(dummy_vec, dummy_map, dummy_goal, action)
    print(f"Eval Value shape: {v.shape}")
    print(f"Eval LogProb shape: {lp.shape}")
    print(f"Eval Entropy shape: {ent.shape}")

    print("Sanity check passed!")