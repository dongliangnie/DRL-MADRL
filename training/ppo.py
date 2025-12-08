"""
Proximal Policy Optimization (PPO) 算法实现
基于论文: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from collections import deque
import warnings


class PPOBuffer:
    """PPO 经验回放缓冲区"""
    
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, 
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        初始化PPO缓冲区
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            buffer_size: 缓冲区大小
            gamma: 折扣因子
            gae_lambda: GAE λ参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 缓冲区
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """
        存储经验
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        
    def finish_path(self, last_value: float = 0):
        """
        完成一个轨迹的计算（计算优势函数和回报）
        
        Args:
            last_value: 轨迹最后状态的价值估计
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # 计算GAE和回报
        rewards = np.append(self.rewards[path_slice], 0)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)
        
        # GAE计算
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        advantages = np.zeros_like(deltas, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(deltas))):
            last_gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        # 计算回报（优势 + 价值）
        returns = advantages + self.values[path_slice]
        
        # 存储优势函数和回报
        self.advantages = advantages if hasattr(self, 'advantages') else np.zeros(self.max_size)
        self.returns = returns if hasattr(self, 'returns') else np.zeros(self.max_size)
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
        
    def get(self) -> Dict[str, np.ndarray]:
        """获取所有经验数据"""
        assert self.ptr == self.max_size, "缓冲区未满"
        
        # 标准化优势函数
        adv_mean = np.mean(self.advantages[:self.ptr])
        adv_std = np.std(self.advantages[:self.ptr])
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=self.states[:self.ptr],
            actions=self.actions[:self.ptr],
            log_probs=self.log_probs[:self.ptr],
            advantages=self.advantages[:self.ptr],
            returns=self.returns[:self.ptr],
            values=self.values[:self.ptr]
        )
        
        # 重置指针
        self.ptr = 0
        self.path_start_idx = 0
        
        return data


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        
        # 输出层：均值和标准差
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            mean: 动作均值
            log_std: 对数标准差
        """
        features = self.network(state)
        mean = torch.tanh(self.mean_layer(features))
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 限制对数标准差范围
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据状态选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性地选择动作
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # 使用重参数化技巧
            log_prob = normal.log_prob(action).sum(dim=-1)
            
            # 确保动作在有效范围内
            action = torch.tanh(action)
            
            # 调整对数概率（由于tanh变换）
            if log_prob is not None:
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
                
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定状态和动作的对数概率和熵
        
        Args:
            state: 状态张量
            action: 动作张量
            
        Returns:
            log_prob: 动作的对数概率
            entropy: 策略熵
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic网络（价值网络）"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            value: 状态价值估计
        """
        return self.network(state).squeeze(-1)


class PPO:
    """Proximal Policy Optimization 算法"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化PPO算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置字典
        """
        # 默认配置
        self.default_config = {
            # 网络结构
            'hidden_dims': [256, 256],
            
            # 训练参数
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            
            # 更新参数
            'ppo_epochs': 10,
            'batch_size': 64,
            'target_kl': 0.01,
            
            # 缓冲区参数
            'buffer_size': 2048,
            
            # 设备
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # 合并配置
        if config is not None:
            self.config = {**self.default_config, **config}
        else:
            self.config = self.default_config
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(self.config['device'])
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim, self.config['hidden_dims']).to(self.device)
        self.critic = CriticNetwork(state_dim, self.config['hidden_dims']).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 经验缓冲区
        self.buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=self.config['buffer_size'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda']
        )
        
        # 训练统计
        self.train_info = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态
            deterministic: 是否确定性地选择动作
            
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor, deterministic)
            
        return action.squeeze(0).cpu().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        存储经验到缓冲区
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 计算动作的对数概率
            _, log_prob = self.actor.get_action(state_tensor, deterministic=False)
            
            # 计算状态价值
            value = self.critic(state_tensor)
            
        # 存储到缓冲区
        self.buffer.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob.item() if log_prob is not None else 0.0,
            value=value.item()
        )
        
    def finish_episode(self, last_value: float = 0):
        """完成一个episode"""
        self.buffer.finish_path(last_value)
        
    def update(self) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Returns:
            info: 训练信息字典
        """
        # 检查缓冲区是否已满
        if self.buffer.ptr < self.buffer.max_size:
            warnings.warn(f"缓冲区未满 ({self.buffer.ptr}/{self.buffer.max_size})，跳过更新")
            return {}
            
        # 获取经验数据
        data = self.buffer.get()
        
        # 转换为张量
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        old_values = torch.FloatTensor(data['values']).to(self.device)
        
        # 训练统计重置
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy_loss = 0
        epoch_approx_kl = 0
        epoch_clip_fraction = 0
        
        # PPO epochs
        for epoch in range(self.config['ppo_epochs']):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            # 小批量更新
            for start in range(0, len(states), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的对数概率和熵
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                
                # 计算比率 (importance sampling)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO Clip 目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 
                                   1 + self.config['clip_ratio']) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                values = self.critic(batch_states)
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # 熵奖励
                entropy_loss = -self.config['entropy_coef'] * entropy.mean()
                
                # 总损失
                total_loss = actor_loss + self.config['value_coef'] * value_loss + entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
                
                self.optimizer.step()
                
                # 记录统计信息
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                
                # 计算近似KL散度
                with torch.no_grad():
                    log_ratio = new_log_probs - batch_old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config['clip_ratio']).float().mean().item()
                    
                epoch_approx_kl += approx_kl
                epoch_clip_fraction += clip_fraction
                
            # 检查KL散度是否过大
            if self.config['target_kl'] is not None and epoch_approx_kl > 1.5 * self.config['target_kl']:
                print(f"提前停止PPO更新，KL散度 {epoch_approx_kl:.4f} 超过阈值")
                break
        
        # 计算平均统计信息
        num_updates = (len(states) // self.config['batch_size']) * self.config['ppo_epochs']
        
        info = {
            'actor_loss': epoch_actor_loss / num_updates,
            'critic_loss': epoch_critic_loss / num_updates,
            'entropy_loss': epoch_entropy_loss / num_updates,
            'approx_kl': epoch_approx_kl / num_updates,
            'clip_fraction': epoch_clip_fraction / num_updates
        }
        
        # 记录到训练信息
        for key, value in info.items():
            self.train_info[key].append(value)
            
        return info
    
    def save_checkpoint(self, filepath: str):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_info': self.train_info
        }
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到 {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_info = checkpoint['train_info']
        
        print(f"从 {filepath} 加载检查点")
        
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return self.train_info
    
    def reset_stats(self):
        """重置训练统计信息"""
        for key in self.train_info:
            self.train_info[key] = []
            

class IPPO(PPO):
    """Independent PPO (IPPO) - 独立PPO，用于多智能体场景"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        初始化IPPO
        
        Args:
            state_dim: 单个智能体的状态维度
            action_dim: 单个智能体的动作维度
            config: 配置字典
        """
        super().__init__(state_dim, action_dim, config)
        
    def update_agents(self, agents_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, float]]:
        """
        更新多个智能体
        
        Args:
            agents_data: 每个智能体的经验数据列表
            
        Returns:
            每个智能体的更新信息列表
        """
        agents_info = []
        
        for i, agent_data in enumerate(agents_data):
            # 将数据存储到缓冲区
            for transition in agent_data['transitions']:
                self.buffer.store(**transition)
                
            # 完成轨迹
            self.buffer.finish_path(agent_data.get('last_value', 0))
            
            # 更新
            info = self.update()
            agents_info.append(info)
            
            # 清空缓冲区（为下一个智能体准备）
            self.buffer = PPOBuffer(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                buffer_size=self.config['buffer_size'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda']
            )
            
        return agents_info


# 工具函数
def compute_gae(next_value: float, rewards: List[float], masks: List[float], 
                values: List[float], gamma: float = 0.99, gae_lambda: float = 0.95) -> List[float]:
    """
    计算广义优势估计 (GAE)
    
    Args:
        next_value: 最后一个状态的价值估计
        rewards: 奖励列表
        masks: 终止标志列表 (1 = 未终止, 0 = 终止)
        values: 价值估计列表
        gamma: 折扣因子
        gae_lambda: GAE λ参数
        
    Returns:
        advantages: 优势函数列表
    """
    values = values + [next_value]
    gae = 0
    advantages = []
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * gae_lambda * masks[t] * gae
        advantages.insert(0, gae)
        
    return advantages


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """
    标准化优势函数
    
    Args:
        advantages: 优势函数数组
        
    Returns:
        标准化后的优势函数
    """
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages)
    
    return (advantages - adv_mean) / (adv_std + 1e-8)


if __name__ == "__main__":
    """测试PPO算法"""
    
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 创建PPO实例
    state_dim = 24
    action_dim = 4
    
    config = {
        'hidden_dims': [64, 64],
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'ppo_epochs': 4,
        'batch_size': 32,
        'buffer_size': 256
    }
    
    ppo = PPO(state_dim, action_dim, config)
    
    print("PPO算法测试:")
    print(f"设备: {ppo.device}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 测试选择动作
    test_state = np.random.randn(state_dim)
    action = ppo.select_action(test_state)
    print(f"测试状态: {test_state[:5]}...")
    print(f"选择的动作: {action}")
    
    # 测试存储经验
    for i in range(10):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i == 9
        
        ppo.store_transition(state, action, reward, next_state, done)
    
    print(f"\n缓冲区大小: {ppo.buffer.ptr}")
    
    # 测试更新
    if ppo.buffer.ptr >= ppo.buffer.max_size:
        info = ppo.update()
        print(f"\n更新完成，损失: {info}")
    else:
        print(f"\n缓冲区未满，需要更多经验")
        
    print("\nPPO算法实现完成！")