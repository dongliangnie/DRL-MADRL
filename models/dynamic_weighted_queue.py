"""
动态加权队列实现
基于论文中的动态加权队列：管理任务分配和优先级更新
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
import heapq
import bisect
from dataclasses import dataclass, field
import time


@dataclass
class QueueItem:
    """队列项"""
    task_id: int
    task_type: str  # 'surveillance' 或 'emergency'
    features: np.ndarray  # 任务特征
    arrival_time: float  # 到达时间
    priority: float = 1.0  # 当前优先级
    estimated_time: float = 0.0  # 估计处理时间
    aoi: float = 0.0  # 当前AoI值
    aoi_threshold: float = 1.0  # AoI阈值
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据
    
    def __post_init__(self):
        self.last_update_time = self.arrival_time
        self.update_count = 0
        
    def update_priority(self, new_priority: float):
        """更新优先级"""
        self.priority = new_priority
        self.last_update_time = time.time()
        self.update_count += 1
        
    def get_urgency(self) -> float:
        """获取紧急程度"""
        # 紧急程度 = 优先级 * (1 / (AoI / AoI阈值))
        aoi_ratio = self.aoi / self.aoi_threshold
        urgency = self.priority * (1.0 / max(aoi_ratio, 0.001))
        return urgency
        
    def is_expired(self, current_time: float, expiry_threshold: float = 3600) -> bool:
        """检查是否过期"""
        return (current_time - self.last_update_time) > expiry_threshold


@dataclass
class QueueConfig:
    """队列配置"""
    max_size: int = 10  # 最大队列长度
    priority_decay: float = 0.95  # 优先级衰减因子
    min_priority: float = 0.1  # 最小优先级
    update_interval: float = 1.0  # 更新间隔（秒）
    
    # 优先级计算参数
    time_weight: float = 0.3  # 时间因素权重
    distance_weight: float = 0.4  # 距离因素权重
    aoi_weight: float = 0.3  # AoI因素权重
    
    # 任务类型权重
    task_type_weights: Dict[str, float] = field(default_factory=lambda: {
        'emergency': 2.0,
        'surveillance': 1.0
    })
    
    # 队列策略
    policy: str = "priority"  # 'priority', 'fifo', 'lifo', 'round_robin'
    enable_dynamic_weights: bool = True  # 是否启用动态权重


class DynamicWeightedQueue:
    """动态加权队列"""
    
    def __init__(self, config: QueueConfig, temporal_predictor=None):
        """
        初始化动态加权队列
        
        Args:
            config: 队列配置
            temporal_predictor: 时间预测器（可选）
        """
        self.config = config
        self.temporal_predictor = temporal_predictor
        
        # 主队列（按优先级排序）
        self.priority_queue = []
        
        # 任务ID到队列项的映射
        self.task_dict = {}
        
        # 历史记录
        self.history = deque(maxlen=1000)
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'processed_tasks': 0,
            'expired_tasks': 0,
            'avg_waiting_time': 0.0,
            'avg_priority': 0.0
        }
        
        # 动态权重
        self.dynamic_weights = {
            'time_weight': config.time_weight,
            'distance_weight': config.distance_weight,
            'aoi_weight': config.aoi_weight
        }
        
        # 最后更新时间
        self.last_update_time = time.time()
        
    def add_task(self, task_id: int, task_type: str, features: np.ndarray, 
                 aoi: float = 0.0, aoi_threshold: float = 1.0, 
                 metadata: Optional[Dict] = None) -> bool:
        """
        添加任务到队列
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            features: 任务特征
            aoi: 当前AoI值
            aoi_threshold: AoI阈值
            metadata: 附加元数据
            
        Returns:
            是否添加成功
        """
        # 检查队列是否已满
        if len(self.priority_queue) >= self.config.max_size:
            # 移除优先级最低的任务
            self._remove_lowest_priority_task()
            
        # 检查任务是否已存在
        if task_id in self.task_dict:
            return False
            
        # 创建队列项
        item = QueueItem(
            task_id=task_id,
            task_type=task_type,
            features=features,
            arrival_time=time.time(),
            aoi=aoi,
            aoi_threshold=aoi_threshold,
            metadata=metadata or {}
        )
        
        # 计算初始优先级
        initial_priority = self._calculate_initial_priority(item)
        item.update_priority(initial_priority)
        
        # 添加到队列
        heapq.heappush(self.priority_queue, (-item.priority, task_id))  # 使用负值实现最大堆
        self.task_dict[task_id] = item
        
        # 更新统计
        self.stats['total_tasks'] += 1
        
        # 记录历史
        self.history.append({
            'time': time.time(),
            'action': 'add',
            'task_id': task_id,
            'task_type': task_type,
            'priority': item.priority
        })
        
        return True
        
    def get_highest_priority_task(self) -> Optional[QueueItem]:
        """
        获取最高优先级的任务
        
        Returns:
            最高优先级任务，如果队列为空则返回None
        """
        if not self.priority_queue:
            return None
            
        # 获取最高优先级任务
        _, task_id = heapq.heappop(self.priority_queue)
        item = self.task_dict.pop(task_id, None)
        
        if item:
            # 更新统计
            self.stats['processed_tasks'] += 1
            waiting_time = time.time() - item.arrival_time
            self.stats['avg_waiting_time'] = (
                self.stats['avg_waiting_time'] * (self.stats['processed_tasks'] - 1) + waiting_time
            ) / self.stats['processed_tasks']
            
            # 记录历史
            self.history.append({
                'time': time.time(),
                'action': 'process',
                'task_id': task_id,
                'waiting_time': waiting_time,
                'priority': item.priority
            })
            
        return item
        
    def peek_highest_priority(self) -> Optional[QueueItem]:
        """
        查看最高优先级的任务（不移除）
        
        Returns:
            最高优先级任务，如果队列为空则返回None
        """
        if not self.priority_queue:
            return None
            
        _, task_id = self.priority_queue[0]
        return self.task_dict.get(task_id)
        
    def update_priorities(self, current_state: np.ndarray, uav_states: List[np.ndarray] = None):
        """
        更新队列中所有任务的优先级
        
        Args:
            current_state: 当前状态
            uav_states: UAV状态列表（可选）
        """
        current_time = time.time()
        
        # 检查是否需要更新
        if current_time - self.last_update_time < self.config.update_interval:
            return
            
        # 更新动态权重（如果启用）
        if self.config.enable_dynamic_weights:
            self._update_dynamic_weights()
            
        # 更新每个任务的优先级
        for task_id, item in list(self.task_dict.items()):
            # 计算新优先级
            new_priority = self._calculate_priority(item, current_state, uav_states)
            
            # 应用衰减
            time_diff = current_time - item.last_update_time
            decay_factor = self.config.priority_decay ** (time_diff / self.config.update_interval)
            new_priority *= decay_factor
            
            # 确保最小优先级
            new_priority = max(new_priority, self.config.min_priority)
            
            # 更新任务
            item.update_priority(new_priority)
            
            # 检查是否过期
            if item.is_expired(current_time):
                self._remove_task(task_id)
                self.stats['expired_tasks'] += 1
                continue
                
        # 重新构建优先队列
        self._rebuild_priority_queue()
        
        # 更新统计
        self._update_statistics()
        
        self.last_update_time = current_time
        
    def _calculate_initial_priority(self, item: QueueItem) -> float:
        """计算初始优先级"""
        # 基础优先级 = 任务类型权重
        base_priority = self.config.task_type_weights.get(item.task_type, 1.0)
        
        # AoI因子
        aoi_factor = 1.0 - min(item.aoi / item.aoi_threshold, 1.0)
        
        # 初始优先级
        priority = base_priority * (0.7 + 0.3 * aoi_factor)
        
        return priority
        
    def _calculate_priority(self, item: QueueItem, current_state: np.ndarray, 
                           uav_states: List[np.ndarray] = None) -> float:
        """
        计算任务优先级
        
        Args:
            item: 队列项
            current_state: 当前状态
            uav_states: UAV状态列表
            
        Returns:
            计算出的优先级
        """
        # 基础优先级
        base_priority = self.config.task_type_weights.get(item.task_type, 1.0)
        
        # 时间因子（任务等待时间）
        current_time = time.time()
        waiting_time = current_time - item.arrival_time
        time_factor = 1.0 / (1.0 + np.log1p(waiting_time))
        
        # AoI因子
        aoi_ratio = item.aoi / item.aoi_threshold
        aoi_factor = 1.0 - min(aoi_ratio, 1.0)
        
        # 距离因子（如果提供了UAV状态）
        distance_factor = 1.0
        if uav_states is not None and len(uav_states) > 0:
            # 计算到最近UAV的距离
            min_distance = float('inf')
            for uav_state in uav_states:
                # 简化距离计算（实际应使用特征向量）
                if len(item.features) == len(uav_state):
                    distance = np.linalg.norm(item.features - uav_state)
                    min_distance = min(min_distance, distance)
                    
            if min_distance < float('inf'):
                # 归一化距离因子
                max_expected_distance = 1000.0  # 假设最大距离
                normalized_distance = min(min_distance / max_expected_distance, 1.0)
                distance_factor = 1.0 - normalized_distance
                
        # 使用时间预测器（如果可用）
        time_prediction_factor = 1.0
        if self.temporal_predictor is not None and hasattr(item, 'estimated_time'):
            try:
                # 预测处理时间
                current_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                goal_tensor = torch.FloatTensor(item.features).unsqueeze(0)
                
                with torch.no_grad():
                    predicted_time = self.temporal_predictor(current_tensor, goal_tensor)
                    
                item.estimated_time = predicted_time.item()
                
                # 时间预测因子
                max_expected_time = 100.0  # 假设最大时间
                normalized_time = min(item.estimated_time / max_expected_time, 1.0)
                time_prediction_factor = 1.0 - normalized_time
                
            except Exception as e:
                # 预测失败，使用默认值
                pass
                
        # 计算综合优先级
        weights = self.dynamic_weights if self.config.enable_dynamic_weights else {
            'time_weight': self.config.time_weight,
            'distance_weight': self.config.distance_weight,
            'aoi_weight': self.config.aoi_weight
        }
        
        priority = base_priority * (
            weights['time_weight'] * time_factor +
            weights['distance_weight'] * distance_factor +
            weights['aoi_weight'] * aoi_factor +
            0.1 * time_prediction_factor  # 时间预测因子
        )
        
        # 应用任务特定调整
        priority = self._apply_task_specific_adjustments(item, priority)
        
        return priority
        
    def _apply_task_specific_adjustments(self, item: QueueItem, priority: float) -> float:
        """应用任务特定调整"""
        # 紧急任务额外加成
        if item.task_type == 'emergency':
            # 根据AoI紧急程度调整
            aoi_urgency = 1.0 - min(item.aoi / item.aoi_threshold, 1.0)
            priority *= 1.0 + 0.5 * aoi_urgency
            
        # 元数据调整
        if 'importance' in item.metadata:
            priority *= item.metadata['importance']
            
        if 'deadline' in item.metadata:
            current_time = time.time()
            time_to_deadline = item.metadata['deadline'] - current_time
            if time_to_deadline > 0:
                # 离截止时间越近，优先级越高
                deadline_factor = 1.0 / (1.0 + time_to_deadline / 60.0)  # 分钟级
                priority *= 1.0 + deadline_factor
                
        return priority
        
    def _update_dynamic_weights(self):
        """更新动态权重"""
        # 基于历史性能调整权重
        if len(self.history) < 10:
            return
            
        # 分析最近的任务处理历史
        recent_history = list(self.history)[-10:]
        processed_count = sum(1 for h in recent_history if h['action'] == 'process')
        
        if processed_count == 0:
            return
            
        # 计算平均等待时间
        waiting_times = [h['waiting_time'] for h in recent_history if 'waiting_time' in h]
        if waiting_times:
            avg_waiting_time = np.mean(waiting_times)
            
            # 基于等待时间调整权重
            if avg_waiting_time > 10.0:  # 等待时间过长
                # 增加时间权重
                self.dynamic_weights['time_weight'] = min(
                    self.dynamic_weights['time_weight'] * 1.1, 0.6
                )
                
    def _remove_lowest_priority_task(self):
        """移除优先级最低的任务"""
        if not self.priority_queue:
            return
            
        # 找到优先级最低的任务
        # 注意：priority_queue是最大堆，所以最低优先级在末尾
        if len(self.priority_queue) > 1:
            # 将堆转换为列表并排序
            sorted_items = sorted(self.priority_queue, key=lambda x: x[0])  # 按优先级排序
            _, lowest_task_id = sorted_items[-1]  # 最低优先级（负值最小）
        else:
            _, lowest_task_id = self.priority_queue[0]
            
        # 移除任务
        self._remove_task(lowest_task_id)
        
    def _remove_task(self, task_id: int):
        """移除指定任务"""
        if task_id in self.task_dict:
            # 从字典中移除
            del self.task_dict[task_id]
            
            # 重建优先队列
            self._rebuild_priority_queue()
            
            # 记录历史
            self.history.append({
                'time': time.time(),
                'action': 'remove',
                'task_id': task_id
            })
            
    def _rebuild_priority_queue(self):
        """重建优先队列"""
        self.priority_queue = []
        for task_id, item in self.task_dict.items():
            heapq.heappush(self.priority_queue, (-item.priority, task_id))
            
    def _update_statistics(self):
        """更新统计信息"""
        if self.task_dict:
            priorities = [item.priority for item in self.task_dict.values()]
            self.stats['avg_priority'] = np.mean(priorities)
        else:
            self.stats['avg_priority'] = 0.0
            
    def get_queue_state(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'size': len(self.priority_queue),
            'max_size': self.config.max_size,
            'tasks': [
                {
                    'task_id': item.task_id,
                    'task_type': item.task_type,
                    'priority': item.priority,
                    'aoi': item.aoi,
                    'aoi_threshold': item.aoi_threshold,
                    'waiting_time': time.time() - item.arrival_time
                }
                for item in self.task_dict.values()
            ],
            'stats': self.stats.copy(),
            'weights': self.dynamic_weights.copy()
        }
        
    def clear(self):
        """清空队列"""
        self.priority_queue.clear()
        self.task_dict.clear()
        
    def __len__(self) -> int:
        """返回队列大小"""
        return len(self.priority_queue)
        
    def __contains__(self, task_id: int) -> bool:
        """检查任务是否在队列中"""
        return task_id in self.task_dict


class MultiUAVQueueManager:
    """多UAV队列管理器"""
    
    def __init__(self, num_uavs: int, config: QueueConfig, temporal_predictor=None):
        """
        初始化多UAV队列管理器
        
        Args:
            num_uavs: UAV数量
            config: 队列配置
            temporal_predictor: 时间预测器
        """
        self.num_uavs = num_uavs
        self.config = config
        
        # 为每个UAV创建一个队列
        self.uav_queues = [DynamicWeightedQueue(config, temporal_predictor) for _ in range(num_uavs)]
        
        # 全局任务分配
        self.global_task_map = {}  # task_id -> uav_id
        
        # 负载均衡统计
        self.load_stats = {
            'queue_lengths': [0] * num_uavs,
            'task_assignments': [0] * num_uavs,
            'last_rebalance_time': time.time()
        }
        
    def assign_task_to_uav(self, task_id: int, task_type: str, features: np.ndarray,
                          uav_states: List[np.ndarray], aoi: float = 0.0,
                          aoi_threshold: float = 1.0, metadata: Optional[Dict] = None) -> int:
        """
        分配任务给UAV
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            features: 任务特征
            uav_states: UAV状态列表
            aoi: AoI值
            aoi_threshold: AoI阈值
            metadata: 元数据
            
        Returns:
            分配的UAV ID，如果失败返回-1
        """
        if len(uav_states) != self.num_uavs:
            raise ValueError(f"UAV状态数量({len(uav_states)})与UAV数量({self.num_uavs})不匹配")
            
        # 检查任务是否已分配
        if task_id in self.global_task_map:
            return self.global_task_map[task_id]
            
        # 选择最佳UAV
        best_uav_id = self._select_best_uav_for_task(features, uav_states, task_type)
        
        if best_uav_id >= 0:
            # 添加到UAV队列
            success = self.uav_queues[best_uav_id].add_task(
                task_id, task_type, features, aoi, aoi_threshold, metadata
            )
            
            if success:
                self.global_task_map[task_id] = best_uav_id
                self.load_stats['task_assignments'][best_uav_id] += 1
                self.load_stats['queue_lengths'][best_uav_id] = len(self.uav_queues[best_uav_id])
                
        return best_uav_id
        
    def _select_best_uav_for_task(self, task_features: np.ndarray, 
                                 uav_states: List[np.ndarray], task_type: str) -> int:
        """选择最适合处理任务的UAV"""
        best_uav_id = -1
        best_score = -float('inf')
        
        for uav_id in range(self.num_uavs):
            # 计算UAV得分
            score = self._calculate_uav_score(uav_id, task_features, uav_states[uav_id], task_type)
            
            if score > best_score:
                best_score = score
                best_uav_id = uav_id
                
        return best_uav_id
        
    def _calculate_uav_score(self, uav_id: int, task_features: np.ndarray, 
                            uav_state: np.ndarray, task_type: str) -> float:
        """计算UAV得分"""
        # 距离得分
        if len(task_features) == len(uav_state):
            distance = np.linalg.norm(task_features - uav_state)
            distance_score = 1.0 / (1.0 + distance)
        else:
            distance_score = 0.5
            
        # 负载得分（队列长度）
        queue_length = len(self.uav_queues[uav_id])
        max_queue_size = self.config.max_size
        load_score = 1.0 - (queue_length / max_queue_size)
        
        # 任务类型匹配得分
        type_score = 1.0
        if task_type == 'emergency':
            # 紧急任务优先分配给负载轻的UAV
            type_score = 2.0 * load_score
            
        # 综合得分
        score = (
            0.4 * distance_score +
            0.4 * load_score +
            0.2 * type_score
        )
        
        return score
        
    def get_task_for_uav(self, uav_id: int, current_state: np.ndarray) -> Optional[QueueItem]:
        """
        获取UAV的下一个任务
        
        Args:
            uav_id: UAV ID
            current_state: 当前状态
            
        Returns:
            下一个任务，如果没有则返回None
        """
        if uav_id < 0 or uav_id >= self.num_uavs:
            return None
            
        queue = self.uav_queues[uav_id]
        
        # 更新队列优先级
        queue.update_priorities(current_state)
        
        # 获取最高优先级任务
        task = queue.get_highest_priority_task()
        
        if task:
            # 从全局映射中移除
            if task.task_id in self.global_task_map:
                del self.global_task_map[task.task_id]
                
            # 更新负载统计
            self.load_stats['queue_lengths'][uav_id] = len(queue)
            
        return task
        
    def rebalance_queues(self, uav_states: List[np.ndarray]):
        """重新平衡队列负载"""
        current_time = time.time()
        
        # 检查是否需要重新平衡
        if current_time - self.load_stats['last_rebalance_time'] < 30.0:  # 30秒间隔
            return
            
        # 计算负载方差
        queue_lengths = [len(q) for q in self.uav_queues]
        load_variance = np.var(queue_lengths)
        
        # 如果负载不平衡，进行重新分配
        if load_variance > 2.0:  # 方差阈值
            self._perform_rebalancing(uav_states)
            
        self.load_stats['last_rebalance_time'] = current_time
        
    def _perform_rebalancing(self, uav_states: List[np.ndarray]):
        """执行重新平衡"""
        # 找到负载最重和最轻的UAV
        queue_lengths = [len(q) for q in self.uav_queues]
        heaviest_uav = np.argmax(queue_lengths)
        lightest_uav = np.argmin(queue_lengths)
        
        # 如果负载差异足够大
        if queue_lengths[heaviest_uav] - queue_lengths[lightest_uav] > 2:
            # 尝试移动任务
            heavy_queue = self.uav_queues[heaviest_uav]
            
            # 获取负载重队列中的所有任务
            tasks_to_consider = []
            for task_id, item in list(heavy_queue.task_dict.items()):
                # 计算任务更适合哪个UAV
                target_uav = self._select_best_uav_for_task(
                    item.features, uav_states, item.task_type
                )
                
                if target_uav == lightest_uav:
                    tasks_to_consider.append((task_id, item))
                    
            # 移动最适合的任务
            if tasks_to_consider:
                # 选择优先级相对较低的任务
                tasks_to_consider.sort(key=lambda x: x[1].priority)
                task_id, item = tasks_to_consider[0]
                
                # 从重负载队列移除
                heavy_queue._remove_task(task_id)
                
                # 添加到轻负载队列
                self.uav_queues[lightest_uav].add_task(
                    task_id, item.task_type, item.features,
                    item.aoi, item.aoi_threshold, item.metadata
                )
                
                # 更新全局映射
                self.global_task_map[task_id] = lightest_uav
                
    def get_manager_state(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'num_uavs': self.num_uavs,
            'queue_lengths': [len(q) for q in self.uav_queues],
            'global_task_count': len(self.global_task_map),
            'load_stats': self.load_stats.copy(),
            'uav_queues': [q.get_queue_state() for q in self.uav_queues]
        }


# 测试函数
def test_dynamic_weighted_queue():
    """测试动态加权队列"""
    print("测试动态加权队列...")
    
    # 创建队列配置
    config = QueueConfig(
        max_size=5,
        priority_decay=0.98,
        min_priority=0.1,
        task_type_weights={'emergency': 2.0, 'surveillance': 1.0}
    )
    
    # 创建队列
    queue = DynamicWeightedQueue(config)
    
    # 添加任务
    print("添加任务...")
    tasks_added = []
    
    for i in range(5):
        task_type = 'emergency' if i % 2 == 0 else 'surveillance'
        features = np.random.randn(10)
        aoi = np.random.uniform(0, 1)
        aoi_threshold = 1.0
        
        success = queue.add_task(
            task_id=i,
            task_type=task_type,
            features=features,
            aoi=aoi,
            aoi_threshold=aoi_threshold,
            metadata={'importance': 1.0 + i * 0.1}
        )
        
        if success:
            tasks_added.append(i)
            print(f"  添加任务 {i} ({task_type}), AoI={aoi:.2f}")
            
    print(f"队列大小: {len(queue)}")
    
    # 获取队列状态
    state = queue.get_queue_state()
    print(f"队列状态: {len(state['tasks'])} 个任务")
    
    # 更新优先级
    print("\n更新优先级...")
    current_state = np.random.randn(10)
    queue.update_priorities(current_state)
    
    # 处理任务
    print("\n处理任务...")
    processed_tasks = []
    
    while len(queue) > 0:
        task = queue.get_highest_priority_task()
        if task:
            processed_tasks.append(task.task_id)
            print(f"  处理任务 {task.task_id} ({task.task_type}), 优先级={task.priority:.3f}")
            
    print(f"共处理 {len(processed_tasks)} 个任务")
    
    # 测试多UAV队列管理器
    print("\n测试多UAV队列管理器...")
    num_uavs = 3
    manager = MultiUAVQueueManager(num_uavs, config)
    
    # 创建UAV状态
    uav_states = [np.random.randn(10) for _ in range(num_uavs)]
    
    # 分配任务
    for i in range(10):
        task_type = 'emergency' if i % 3 == 0 else 'surveillance'
        features = np.random.randn(10)
        aoi = np.random.uniform(0, 1)
        
        uav_id = manager.assign_task_to_uav(
            task_id=100 + i,
            task_type=task_type,
            features=features,
            uav_states=uav_states,
            aoi=aoi,
            aoi_threshold=1.0
        )
        
        if uav_id >= 0:
            print(f"  任务 {100 + i} 分配给 UAV {uav_id}")
            
    # 获取管理器状态
    manager_state = manager.get_manager_state()
    print(f"管理器状态: {manager_state['queue_lengths']}")
    
    # 模拟UAV获取任务
    print("\n模拟UAV获取任务...")
    for uav_id in range(num_uavs):
        current_state = np.random.randn(10)
        task = manager.get_task_for_uav(uav_id, current_state)
        
        if task:
            print(f"  UAV {uav_id} 获取任务 {task.task_id}")
        else:
            print(f"  UAV {uav_id} 没有任务")
            
    print("\n动态加权队列测试完成!")


if __name__ == "__main__":
    test_dynamic_weighted_queue()