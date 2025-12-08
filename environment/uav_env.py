"""
多任务导向的应急感知UAV群智感知环境
基于论文 "Multi-Task-Oriented Emergency-Aware UAV Crowdsensing: A Hierarchical Multi-Agent Deep Reinforcement Learning Approach"
"""

import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import pandas as pd
from collections import deque

# 设置随机种子
np.random.seed(42)
random.seed(42)


class TaskType(Enum):
    """任务类型枚举"""
    SURVEILLANCE = 0  # 监控任务
    EMERGENCY = 1     # 紧急任务


@dataclass
class Task:
    """任务基类"""
    task_id: int
    task_type: TaskType
    location: np.ndarray  # [x, y] 坐标
    aoi: int  # 信息年龄
    aoi_threshold: int  # AoI阈值
    data_size: float  # 数据大小 (MB)
    priority: float = 1.0  # 任务优先级
    
    def __post_init__(self):
        self.handled = False
        self.handling_time = 0
        self.last_handled_time = 0
        
    def update_aoi(self, current_time: int):
        """更新AoI"""
        if self.handled:
            self.aoi = 1
            self.handled = False
        else:
            self.aoi = current_time - self.last_handled_time + 1
            
    def is_valid(self) -> bool:
        """检查任务是否有效（AoI未超过阈值）"""
        return self.aoi <= self.aoi_threshold
    
    def get_aoi_normalized(self) -> float:
        """获取归一化的AoI"""
        return min(self.aoi / self.aoi_threshold, 1.0)


@dataclass
class SurveillanceTask(Task):
    """监控任务"""
    def __init__(self, task_id: int, location: np.ndarray, aoi_threshold: int = 35, 
                 data_size: float = 10.0):
        super().__init__(task_id, TaskType.SURVEILLANCE, location, 0, 
                        aoi_threshold, data_size)
        self.data_generation_rate = 1.0  # MB/timeslot
        self.current_data = data_size
        
    def generate_data(self):
        """生成新数据"""
        self.current_data += self.data_generation_rate
        
    def collect_data(self, collection_rate: float) -> float:
        """收集数据，返回实际收集的数据量"""
        collected = min(collection_rate, self.current_data)
        self.current_data -= collected
        self.handled = True
        return collected


@dataclass
class EmergencyTask(Task):
    """紧急任务"""
    def __init__(self, task_id: int, location: np.ndarray, aoi_threshold: int = 20,
                 data_size: float = 50.0, area_size: float = 100.0):
        super().__init__(task_id, TaskType.EMERGENCY, location, 0, 
                        aoi_threshold, data_size, priority=2.0)
        self.area_size = area_size  # 紧急区域大小 (m²)
        self.images_needed = 10  # 需要的图像数量
        self.images_captured = 0
        self.capture_progress = 0.0  # 0-1，完成百分比
        
    def capture_image(self, image_quality: float) -> float:
        """捕获图像，返回图像质量分数"""
        if self.images_captured < self.images_needed:
            self.images_captured += 1
            self.capture_progress = self.images_captured / self.images_needed
            if self.images_captured >= self.images_needed:
                self.handled = True
            return image_quality
        return 0.0


@dataclass
class UAV:
    """无人机类"""
    uav_id: int
    initial_location: np.ndarray
    max_energy: float = 10000.0  # 最大能量 (J)
    max_speed: float = 20.0  # 最大速度 (m/s)
    communication_range: float = 500.0  # 通信范围 (m)
    sensing_range: float = 200.0  # 传感范围 (m)
    
    def __post_init__(self):
        self.location = self.initial_location.copy()
        self.energy = self.max_energy
        self.velocity = np.zeros(2)
        self.current_task = None
        self.task_queue = deque(maxlen=5)  # 任务队列
        self.collected_data = 0.0
        self.images_captured = 0
        self.history = []  # 历史轨迹
        
    def move(self, action: np.ndarray, dt: float = 1.0):
        """
        移动UAV
        
        Args:
            action: [v_x, v_y] 速度向量，归一化到[-1, 1]
            dt: 时间步长 (s)
        """
        # 限制速度大小
        speed = np.linalg.norm(action)
        if speed > 1.0:
            action = action / speed
            
        # 计算实际速度
        self.velocity = action * self.max_speed
        
        # 更新位置
        new_location = self.location + self.velocity * dt
        
        # 计算能量消耗
        energy_consumption = self.calculate_energy_consumption(self.velocity, dt)
        
        # 检查能量
        if self.energy >= energy_consumption:
            self.location = new_location
            self.energy -= energy_consumption
            self.history.append(self.location.copy())
        else:
            # 能量不足，只能移动部分距离
            factor = self.energy / energy_consumption
            self.location = self.location + self.velocity * dt * factor
            self.energy = 0
            
        return energy_consumption
    
    def calculate_energy_consumption(self, velocity: np.ndarray, dt: float) -> float:
        """计算能量消耗"""
        # 简化模型：能量消耗与速度平方成正比
        speed = np.linalg.norm(velocity)
        return 0.1 * speed**2 * dt + 0.5 * dt  # 基础功耗
    
    def distance_to(self, target: np.ndarray) -> float:
        """计算到目标点的距离"""
        return np.linalg.norm(self.location - target)
    
    def can_communicate(self, task_location: np.ndarray) -> bool:
        """检查是否可以与任务点通信"""
        return self.distance_to(task_location) <= self.communication_range
    
    def can_sense(self, task_location: np.ndarray) -> bool:
        """检查是否可以感知任务点"""
        return self.distance_to(task_location) <= self.sensing_range
    
    def get_state(self) -> Dict[str, Any]:
        """获取UAV状态"""
        return {
            'id': self.uav_id,
            'location': self.location.copy(),
            'energy': self.energy,
            'velocity': self.velocity.copy(),
            'current_task': self.current_task.task_id if self.current_task else -1,
            'queue_size': len(self.task_queue)
        }
    
    def reset(self):
        """重置UAV状态"""
        self.location = self.initial_location.copy()
        self.energy = self.max_energy
        self.velocity = np.zeros(2)
        self.current_task = None
        self.task_queue.clear()
        self.collected_data = 0.0
        self.images_captured = 0
        self.history = [self.location.copy()]


class CommunicationChannel:
    """通信信道模型"""
    
    def __init__(self, frequency: float = 2.4e9, bandwidth: float = 10e6, 
                 noise_power: float = -174, tx_power: float = 20):
        """
        初始化通信信道
        
        Args:
            frequency: 载波频率 (Hz)
            bandwidth: 带宽 (Hz)
            noise_power: 噪声功率 (dBm)
            tx_power: 发射功率 (dBm)
        """
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.noise_power = noise_power
        self.tx_power = tx_power
        self.speed_of_light = 3e8
        
        # 信道参数（基于3GPP模型）
        self.alpha_1 = 9.61  # 城市环境参数
        self.alpha_2 = 0.16
        self.path_loss_los = 2.0  # 视距路径损耗指数
        self.path_loss_nlos = 3.5  # 非视距路径损耗指数
        self.additional_loss_los = 2.0  # dB
        self.additional_loss_nlos = 23.0  # dB
        
    def calculate_path_loss(self, distance: float, is_los: bool = True) -> float:
        """计算路径损耗"""
        if is_los:
            # 自由空间路径损耗 + 额外损耗
            fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55
            return fspl + self.additional_loss_los
        else:
            # NLOS路径损耗
            return self.path_loss_nlos * 10 * np.log10(distance) + self.additional_loss_nlos
    
    def calculate_los_probability(self, distance: float, height_difference: float = 50) -> float:
        """计算视距概率"""
        # 基于高度的简单模型
        if height_difference <= 0:
            return 0.0
            
        elevation_angle = np.degrees(np.arctan(height_difference / distance))
        return 1.0 / (1 + self.alpha_1 * np.exp(-self.alpha_2 * (elevation_angle - self.alpha_1)))
    
    def calculate_data_rate(self, distance: float, height_difference: float = 50) -> float:
        """计算数据传输速率 (bps)"""
        # 计算视距概率
        p_los = self.calculate_los_probability(distance, height_difference)
        
        # 计算平均路径损耗
        pl_los = self.calculate_path_loss(distance, is_los=True)
        pl_nlos = self.calculate_path_loss(distance, is_los=False)
        avg_path_loss = p_los * pl_los + (1 - p_los) * pl_nlos
        
        # 计算接收信号强度
        rx_power = self.tx_power - avg_path_loss
        
        # 计算信噪比
        snr = rx_power - self.noise_power
        
        # 计算香农容量
        if snr <= 0:
            return 0.0
            
        data_rate = self.bandwidth * np.log2(1 + 10**(snr/10))
        
        return data_rate


class CameraModel:
    """相机模型"""
    
    def __init__(self, focal_length: float = 35.0, sensor_width: float = 23.5, 
                 pixel_size: float = 5.86e-6, exposure_time: float = 1/1000):
        """
        初始化相机模型
        
        Args:
            focal_length: 焦距 (mm)
            sensor_width: 传感器宽度 (mm)
            pixel_size: 像素尺寸 (m)
            exposure_time: 曝光时间 (s)
        """
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.pixel_size = pixel_size
        self.exposure_time = exposure_time
        
        # 计算地面采样距离
        self.gsd_at_100m = self.calculate_gsd(100)  # 100米高度的GSD
        
    def calculate_gsd(self, altitude: float) -> float:
        """计算地面采样距离 (Ground Sampling Distance)"""
        # GSD = (像素尺寸 * 高度) / 焦距
        return (self.pixel_size * altitude) / (self.focal_length * 1e-3)
    
    def calculate_image_blur(self, uav_speed: float, altitude: float) -> float:
        """
        计算图像模糊
        
        Args:
            uav_speed: UAV速度 (m/s)
            altitude: 高度 (m)
            
        Returns:
            模糊像素数
        """
        # 模糊 = (速度 * 曝光时间) / GSD
        gsd = self.calculate_gsd(altitude)
        if gsd == 0:
            return float('inf')
            
        blur_pixels = (uav_speed * self.exposure_time) / gsd
        return blur_pixels
    
    def calculate_coverage_area(self, altitude: float) -> float:
        """计算覆盖面积"""
        # 覆盖宽度 = (传感器宽度 * 高度) / 焦距
        coverage_width = (self.sensor_width * 1e-3 * altitude) / (self.focal_length * 1e-3)
        coverage_area = coverage_width ** 2  # 假设正方形覆盖
        return coverage_area


class UAVEnvironment:
    """多UAV多任务环境"""
    
    def __init__(self, 
                 map_size: Tuple[float, float] = (1000, 1000),
                 num_uavs: int = 4,
                 num_surveillance_tasks: int = 20,
                 max_emergency_tasks: int = 5,
                 time_slot_duration: float = 20.0,
                 max_time_slots: int = 100,
                 config: Optional[Dict] = None):
        """
        初始化UAV环境
        
        Args:
            map_size: 地图大小 (宽, 高) (m)
            num_uavs: UAV数量
            num_surveillance_tasks: 监控任务数量
            max_emergency_tasks: 最大同时紧急任务数量
            time_slot_duration: 时隙持续时间 (s)
            max_time_slots: 最大时隙数
            config: 配置字典
        """
        # 环境参数
        self.map_width, self.map_height = map_size
        self.num_uavs = num_uavs
        self.num_surveillance_tasks = num_surveillance_tasks
        self.max_emergency_tasks = max_emergency_tasks
        self.time_slot_duration = time_slot_duration
        self.max_time_slots = max_time_slots
        
        # 合并配置
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
            
        # 初始化组件
        self.communication_channel = CommunicationChannel()
        self.camera_model = CameraModel()
        
        # 环境状态
        self.uavs = []
        self.surveillance_tasks = []
        self.emergency_tasks = []
        self.all_tasks = []
        
        self.current_time_slot = 0
        self.total_reward = 0.0
        self.done = False
        
        # 统计信息
        self.stats = {
            'surveillance_handled': 0,
            'emergency_handled': 0,
            'surveillance_violated': 0,
            'emergency_violated': 0,
            'total_energy_consumed': 0.0,
            'total_data_collected': 0.0,
            'total_images_captured': 0
        }
        
        # 任务ID计数器
        self.next_task_id = 0
        
        # 初始化环境
        self._initialize_environment()
        
        # 观察和动作空间
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # 可视化设置
        self.fig = None
        self.ax = None
        
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            # UAV参数
            'uav_max_energy': 10000.0,
            'uav_max_speed': 20.0,
            'uav_communication_range': 500.0,
            'uav_sensing_range': 200.0,
            
            # 任务参数
            'surveillance_aoi_threshold': 35,
            'emergency_aoi_threshold': 20,
            'emergency_area_size': 100.0,
            'surveillance_data_size': 10.0,
            'emergency_data_size': 50.0,
            
            # 任务生成
            'emergency_probability': 0.1,
            'emergency_interval': 6,
            
            # 奖励参数
            'reward_surveillance_handled': 1.0,
            'reward_emergency_handled': 5.0,
            'penalty_aoi_violation': -2.0,
            'penalty_energy_consumption': -0.01,
            'penalty_collision': -10.0,
            
            # 其他参数
            'collision_distance': 10.0,
            'image_blur_threshold': 5.0,
            'uav_altitude': 100.0
        }
        
    def _initialize_environment(self):
        """初始化环境"""
        # 重置状态
        self.uavs = []
        self.surveillance_tasks = []
        self.emergency_tasks = []
        self.all_tasks = []
        self.next_task_id = 0
        
        # 生成UAV
        self._generate_uavs()
        
        # 生成监控任务
        self._generate_surveillance_tasks()
        
        # 生成初始紧急任务
        self._generate_initial_emergency_tasks()
        
        # 合并所有任务
        self.all_tasks = self.surveillance_tasks + self.emergency_tasks
        
        # 重置时间和统计
        self.current_time_slot = 0
        self.total_reward = 0.0
        self.done = False
        self.stats = {k: 0 for k in self.stats.keys()}
        
    def _generate_uavs(self):
        """生成UAV"""
        # 在地图上均匀分布UAV
        for i in range(self.num_uavs):
            # 计算位置
            x = (i % 2 + 0.5) * (self.map_width / 2)
            y = (i // 2 + 0.5) * (self.map_height / 2)
            location = np.array([x, y])
            
            # 创建UAV
            uav = UAV(
                uav_id=i,
                initial_location=location,
                max_energy=self.config['uav_max_energy'],
                max_speed=self.config['uav_max_speed'],
                communication_range=self.config['uav_communication_range'],
                sensing_range=self.config['uav_sensing_range']
            )
            
            self.uavs.append(uav)
            
    def _generate_surveillance_tasks(self):
        """生成监控任务"""
        # 在地图上随机分布监控任务
        for i in range(self.num_surveillance_tasks):
            # 随机位置
            x = np.random.uniform(0, self.map_width)
            y = np.random.uniform(0, self.map_height)
            location = np.array([x, y])
            
            # 创建监控任务
            task = SurveillanceTask(
                task_id=self.next_task_id,
                location=location,
                aoi_threshold=self.config['surveillance_aoi_threshold'],
                data_size=self.config['surveillance_data_size']
            )
            
            self.surveillance_tasks.append(task)
            self.next_task_id += 1
            
    def _generate_initial_emergency_tasks(self):
        """生成初始紧急任务"""
        # 生成1-2个初始紧急任务
        num_initial = np.random.randint(1, 3)
        
        for i in range(num_initial):
            if len(self.emergency_tasks) >= self.max_emergency_tasks:
                break
                
            # 随机位置
            x = np.random.uniform(0, self.map_width)
            y = np.random.uniform(0, self.map_height)
            location = np.array([x, y])
            
            # 创建紧急任务
            task = EmergencyTask(
                task_id=self.next_task_id,
                location=location,
                aoi_threshold=self.config['emergency_aoi_threshold'],
                data_size=self.config['emergency_data_size'],
                area_size=self.config['emergency_area_size']
            )
            
            self.emergency_tasks.append(task)
            self.all_tasks.append(task)
            self.next_task_id += 1
            
    def _generate_new_emergency_task(self):
        """生成新的紧急任务"""
        if len(self.emergency_tasks) >= self.max_emergency_tasks:
            return False
            
        # 检查是否应该生成新任务
        if self.current_time_slot % self.config['emergency_interval'] == 0:
            # 随机位置（避免与现有任务太近）
            max_attempts = 10
            for _ in range(max_attempts):
                x = np.random.uniform(0, self.map_width)
                y = np.random.uniform(0, self.map_height)
                location = np.array([x, y])
                
                # 检查距离
                too_close = False
                for task in self.emergency_tasks:
                    if np.linalg.norm(location - task.location) < 100:
                        too_close = True
                        break
                        
                if not too_close:
                    # 创建紧急任务
                    task = EmergencyTask(
                        task_id=self.next_task_id,
                        location=location,
                        aoi_threshold=self.config['emergency_aoi_threshold'],
                        data_size=self.config['emergency_data_size'],
                        area_size=self.config['emergency_area_size']
                    )
                    
                    self.emergency_tasks.append(task)
                    self.all_tasks.append(task)
                    self.next_task_id += 1
                    return True
                    
        return False
        
    def _get_observation_space(self) -> Dict:
        """获取观察空间"""
        # 每个UAV的观察维度
        uav_obs_dim = 4 + self.num_surveillance_tasks + self.max_emergency_tasks * 3
        
        # 全局观察维度（用于集中式训练）
        global_obs_dim = (self.num_uavs * 4 + 
                         self.num_surveillance_tasks * 2 + 
                         self.max_emergency_tasks * 4)
        
        return {
            'uav_observation_dim': uav_obs_dim,
            'global_observation_dim': global_obs_dim,
            'uav_action_dim': 2,  # [v_x, v_y]
            'high_level_action_dim': self.num_uavs,  # 分配任务给哪个UAV
            'num_uavs': self.num_uavs,
            'num_surveillance_tasks': self.num_surveillance_tasks,
            'max_emergency_tasks': self.max_emergency_tasks
        }
        
    def _get_action_space(self) -> Dict:
        """获取动作空间"""
        return {
            'low_level': {
                'type': 'continuous',
                'shape': (2,),  # 速度向量
                'low': -1.0,
                'high': 1.0
            },
            'high_level': {
                'type': 'discrete',
                'shape': (),
                'n': self.num_uavs  # 选择UAV编号
            }
        }
        
    def get_uav_observation(self, uav_id: int) -> np.ndarray:
        """获取单个UAV的观察"""
        uav = self.uavs[uav_id]
        
        # 基本状态
        obs = [
            uav.location[0] / self.map_width,  # 归一化x坐标
            uav.location[1] / self.map_height, # 归一化y坐标
            uav.energy / self.config['uav_max_energy'],  # 归一化能量
            uav.velocity[0] / self.config['uav_max_speed'],  # 归一化速度x
            uav.velocity[1] / self.config['uav_max_speed']   # 归一化速度y
        ]
        
        # 监控任务信息（归一化距离和AoI）
        for task in self.surveillance_tasks:
            distance = uav.distance_to(task.location)
            normalized_distance = min(distance / self.map_width, 1.0)
            normalized_aoi = task.get_aoi_normalized()
            
            obs.append(normalized_distance)
            obs.append(normalized_aoi)
            
        # 紧急任务信息
        for i in range(self.max_emergency_tasks):
            if i < len(self.emergency_tasks):
                task = self.emergency_tasks[i]
                distance = uav.distance_to(task.location)
                normalized_distance = min(distance / self.map_width, 1.0)
                normalized_aoi = task.get_aoi_normalized()
                normalized_progress = task.capture_progress
                
                obs.append(normalized_distance)
                obs.append(normalized_aoi)
                obs.append(normalized_progress)
            else:
                # 填充空任务信息
                obs.extend([1.0, 1.0, 0.0])
                
        return np.array(obs, dtype=np.float32)
        
    def get_global_observation(self) -> np.ndarray:
        """获取全局观察（用于集中式训练）"""
        obs = []
        
        # UAV状态
        for uav in self.uavs:
            obs.extend([
                uav.location[0] / self.map_width,
                uav.location[1] / self.map_height,
                uav.energy / self.config['uav_max_energy'],
                len(uav.task_queue) / 5.0  # 归一化队列长度
            ])
            
        # 监控任务状态
        for task in self.surveillance_tasks:
            obs.extend([
                task.location[0] / self.map_width,
                task.location[1] / self.map_height,
                task.get_aoi_normalized()
            ])
            
        # 紧急任务状态
        for i in range(self.max_emergency_tasks):
            if i < len(self.emergency_tasks):
                task = self.emergency_tasks[i]
                obs.extend([
                    task.location[0] / self.map_width,
                    task.location[1] / self.map_height,
                    task.get_aoi_normalized(),
                    task.capture_progress
                ])
            else:
                obs.extend([0.0, 0.0, 1.0, 0.0])
                
        return np.array(obs, dtype=np.float32)
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        self._initialize_environment()
        
        # 获取初始观察
        uav_observations = [self.get_uav_observation(i) for i in range(self.num_uavs)]
        global_observation = self.get_global_observation()
        
        # 信息字典
        info = {
            'uav_states': [uav.get_state() for uav in self.uavs],
            'task_states': [task.__dict__ for task in self.all_tasks],
            'stats': self.stats.copy(),
            'current_time_slot': self.current_time_slot
        }
        
        return uav_observations, global_observation, info
        
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, List[float], bool, Dict]:
        """
        执行一步环境更新
        
        Args:
            actions: 每个UAV的动作列表，每个动作是归一化的速度向量
            
        Returns:
            uav_observations: 每个UAV的新观察
            global_observation: 全局观察
            rewards: 每个UAV的奖励
            done: 是否结束
            info: 信息字典
        """
        # 更新UAV位置
        total_energy_consumed = 0.0
        
        for i, action in enumerate(actions):
            if i < len(self.uavs):
                energy = self.uavs[i].move(action, self.time_slot_duration)
                total_energy_consumed += energy
                
        # 更新统计
        self.stats['total_energy_consumed'] += total_energy_consumed
        
        # 处理任务
        self._handle_tasks()
        
        # 更新AoI
        self._update_aoi()
        
        # 生成新紧急任务
        self._generate_new_emergency_task()
        
        # 更新时间
        self.current_time_slot += 1
        
        # 检查是否结束
        self.done = (self.current_time_slot >= self.max_time_slots) or \
                   all(uav.energy <= 0 for uav in self.uavs)
        
        # 计算奖励
        rewards = self._calculate_rewards()
        
        # 获取新观察
        uav_observations = [self.get_uav_observation(i) for i in range(self.num_uavs)]
        global_observation = self.get_global_observation()
        
        # 信息字典
        info = {
            'uav_states': [uav.get_state() for uav in self.uavs],
            'task_states': [task.__dict__ for task in self.all_tasks],
            'stats': self.stats.copy(),
            'current_time_slot': self.current_time_slot,
            'total_reward': sum(rewards)
        }
        
        return uav_observations, global_observation, rewards, self.done, info
        
    def _handle_tasks(self):
        """处理任务"""
        for uav in self.uavs:
            # 处理监控任务
            for task in self.surveillance_tasks:
                if uav.can_sense(task.location) and task.is_valid():
                    # 计算数据传输速率
                    distance = uav.distance_to(task.location)
                    data_rate = self.communication_channel.calculate_data_rate(
                        distance, self.config['uav_altitude'])
                    
                    # 收集数据
                    collected = task.collect_data(data_rate * self.time_slot_duration)
                    if collected > 0:
                        uav.collected_data += collected
                        self.stats['total_data_collected'] += collected
                        self.stats['surveillance_handled'] += 1
                        
            # 处理紧急任务
            for task in self.emergency_tasks:
                if uav.can_sense(task.location) and task.is_valid():
                    # 计算图像质量（考虑模糊）
                    uav_speed = np.linalg.norm(uav.velocity)
                    blur = self.camera_model.calculate_image_blur(
                        uav_speed, self.config['uav_altitude'])
                    
                    # 如果模糊在阈值内，捕获图像
                    if blur <= self.config['image_blur_threshold']:
                        image_quality = max(0, 1 - blur / self.config['image_blur_threshold'])
                        quality_score = task.capture_image(image_quality)
                        
                        if quality_score > 0:
                            uav.images_captured += 1
                            self.stats['total_images_captured'] += 1
                            
                            if task.handled:
                                self.stats['emergency_handled'] += 1
                                
    def _update_aoi(self):
        """更新所有任务的AoI"""
        # 更新监控任务
        for task in self.surveillance_tasks:
            task.update_aoi(self.current_time_slot)
            if not task.is_valid():
                task.last_handled_time = self.current_time_slot
                self.stats['surveillance_violated'] += 1
                
        # 更新紧急任务
        for task in self.emergency_tasks:
            task.update_aoi(self.current_time_slot)
            if not task.is_valid():
                task.last_handled_time = self.current_time_slot
                self.stats['emergency_violated'] += 1
                
        # 移除已完成或超时的紧急任务
        self.emergency_tasks = [
            task for task in self.emergency_tasks 
            if task.is_valid() or task.handled
        ]
        
    def _calculate_rewards(self) -> List[float]:
        """计算每个UAV的奖励"""
        rewards = []
        
        for uav in self.uavs:
            reward = 0.0
            
            # 任务处理奖励（通过统计信息间接计算）
            reward += self.stats['surveillance_handled'] * self.config['reward_surveillance_handled']
            reward += self.stats['emergency_handled'] * self.config['reward_emergency_handled']
            
            # AoI违规惩罚
            reward += self.stats['surveillance_violated'] * self.config['penalty_aoi_violation']
            reward += self.stats['emergency_violated'] * self.config['penalty_aoi_violation']
            
            # 能量消耗惩罚
            reward += self.stats['total_energy_consumed'] * self.config['penalty_energy_consumption']
            
            # 碰撞检查（简化）
            for other_uav in self.uavs:
                if uav.uav_id != other_uav.uav_id:
                    distance = uav.distance_to(other_uav.location)
                    if distance < self.config['collision_distance']:
                        reward += self.config['penalty_collision']
                        
            rewards.append(reward)
            
        # 归一化奖励
        if rewards:
            max_abs_reward = max(abs(r) for r in rewards)
            if max_abs_reward > 0:
                rewards = [r / max_abs_reward for r in rewards]
                
        return rewards
        
    def get_valid_task_handling_index(self) -> float:
        """计算有效任务处理指数"""
        # 计算监控任务有效处理率
        if self.num_surveillance_tasks > 0:
            surveillance_valid = sum(1 for task in self.surveillance_tasks if task.is_valid())
            surveillance_ratio = surveillance_valid / self.num_surveillance_tasks
        else:
            surveillance_ratio = 1.0
            
        # 计算紧急任务有效处理率
        if len(self.emergency_tasks) > 0:
            emergency_valid = sum(1 for task in self.emergency_tasks if task.is_valid() or task.handled)
            emergency_ratio = emergency_valid / len(self.emergency_tasks)
        else:
            emergency_ratio = 1.0
            
        # 计算能量消耗率
        total_max_energy = self.num_uavs * self.config['uav_max_energy']
        if total_max_energy > 0:
            energy_ratio = self.stats['total_energy_consumed'] / total_max_energy
        else:
            energy_ratio = 0.0
            
        # 有效任务处理指数 = min(监控任务处理率, 紧急任务处理率) / 能量消耗率
        if energy_ratio > 0:
            valid_index = min(surveillance_ratio, emergency_ratio) / energy_ratio
        else:
            valid_index = min(surveillance_ratio, emergency_ratio)
            
        return valid_index
        
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            
        self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'UAV Environment - Time Slot: {self.current_time_slot}')
        
        # 绘制监控任务点
        for task in self.surveillance_tasks:
            color = 'green' if task.is_valid() else 'red'
            alpha = max(0.3, 1.0 - task.get_aoi_normalized())
            
            self.ax.scatter(task.location[0], task.location[1], 
                          c=color, s=100, alpha=alpha, marker='s', 
                          label='Surveillance' if task.task_id == 0 else None)
            
            # 显示AoI
            self.ax.text(task.location[0], task.location[1] + 20, 
                       f'AoI: {task.aoi}', fontsize=8, ha='center')
                       
        # 绘制紧急任务点
        for task in self.emergency_tasks:
            color = 'orange' if task.is_valid() else 'red'
            alpha = max(0.5, 1.0 - task.get_aoi_normalized())
            
            # 绘制区域
            area_size = task.area_size
            rect = Rectangle((task.location[0] - area_size/2, task.location[1] - area_size/2),
                           area_size, area_size, linewidth=2, 
                           edgecolor=color, facecolor='none', alpha=alpha)
            self.ax.add_patch(rect)
            
            # 显示进度
            self.ax.text(task.location[0], task.location[1], 
                       f'{task.capture_progress*100:.0f}%', 
                       fontsize=9, ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            if task.task_id == self.emergency_tasks[0].task_id:
                self.ax.add_patch(rect)
                self.ax.text(task.location[0], task.location[1] + area_size/2 + 20,
                           'Emergency', fontsize=9, ha='center')
                           
        # 绘制UAV
        for uav in self.uavs:
            # UAV位置
            self.ax.scatter(uav.location[0], uav.location[1], 
                          c='blue', s=200, marker='^', 
                          label='UAV' if uav.uav_id == 0 else None)
            
            # 显示UAV ID和能量
            self.ax.text(uav.location[0], uav.location[1] + 30,
                       f'UAV {uav.uav_id}\nE: {uav.energy:.0f}', 
                       fontsize=8, ha='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # 绘制轨迹
            if len(uav.history) > 1:
                history = np.array(uav.history)
                self.ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.5, linewidth=1)
                
            # 绘制传感范围
            circle = Circle((uav.location[0], uav.location[1]), 
                          self.config['uav_sensing_range'], 
                          fill=False, linestyle='--', alpha=0.3, color='blue')
            self.ax.add_patch(circle)
            
        # 显示统计信息
        stats_text = (
            f"Valid Task Handling Index: {self.get_valid_task_handling_index():.2f}\n"
            f"Surveillance Handled: {self.stats['surveillance_handled']}\n"
            f"Emergency Handled: {self.stats['emergency_handled']}\n"
            f"Data Collected: {self.stats['total_data_collected']:.1f} MB\n"
            f"Images Captured: {self.stats['total_images_captured']}\n"
            f"Energy Consumed: {self.stats['total_energy_consumed']:.0f} J"
        )
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
                    
        # 图例
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles[:3], labels[:3], loc='upper right')
            
        plt.tight_layout()
        
        if mode == 'human':
            plt.pause(0.1)
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image
            
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
    def get_environment_info(self) -> Dict:
        """获取环境信息"""
        return {
            'config': self.config,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'num_uavs': self.num_uavs,
            'num_surveillance_tasks': self.num_surveillance_tasks,
            'current_emergency_tasks': len(self.emergency_tasks),
            'current_time_slot': self.current_time_slot,
            'valid_task_handling_index': self.get_valid_task_handling_index(),
            'stats': self.stats.copy()
        }


# 测试环境
def test_environment():
    """测试UAV环境"""
    print("测试UAV环境...")
    
    # 创建环境
    env = UAVEnvironment(
        map_size=(1000, 1000),
        num_uavs=4,
        num_surveillance_tasks=10,
        max_emergency_tasks=3,
        time_slot_duration=20.0,
        max_time_slots=50
    )
    
    # 重置环境
    uav_obs, global_obs, info = env.reset()
    
    print(f"环境创建成功!")
    print(f"UAV数量: {env.num_uavs}")
    print(f"监控任务数量: {env.num_surveillance_tasks}")
    print(f"紧急任务数量: {len(env.emergency_tasks)}")
    print(f"UAV观察维度: {len(uav_obs[0])}")
    print(f"全局观察维度: {len(global_obs)}")
    
    # 运行几个时间步
    max_steps = 10
    total_reward = 0
    
    for step in range(max_steps):
        print(f"\n--- 时间步 {step} ---")
        
        # 随机动作
        actions = [np.random.uniform(-1, 1, 2) for _ in range(env.num_uavs)]
        
        # 执行一步
        uav_obs, global_obs, rewards, done, info = env.step(actions)
        
        # 计算总奖励
        step_reward = sum(rewards)
        total_reward += step_reward
        
        print(f"奖励: {rewards}")
        print(f"UAV能量: {[uav.energy for uav in env.uavs]}")
        print(f"有效任务处理指数: {env.get_valid_task_handling_index():.2f}")
        
        # 渲染环境
        if step % 2 == 0:
            env.render()
            
        if done:
            print("环境结束!")
            break
            
    print(f"\n总奖励: {total_reward:.2f}")
    
    # 显示统计信息
    print("\n最终统计:")
    stats = env.get_environment_info()['stats']
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    env.close()
    print("\n环境测试完成!")
    

# if __name__ == "__main__":
#     test_environment()
# 在文件末尾添加这个测试部分
if __name__ == "__main__":
    import sys
    import traceback
    
    print("=" * 50)
    print("Starting UAV Environment Test...")
    print("=" * 50)
    
    try:
        # 测试环境创建
        print("\n1. Testing environment creation...")
        env = UAVEnvironment(
            map_size=(800, 800),  # 使用更小的地图以加快测试
            num_uavs=2,           # 减少UAV数量
            num_surveillance_tasks=5,  # 减少监控任务
            max_emergency_tasks=2,      # 减少紧急任务
            time_slot_duration=10.0,
            max_time_slots=10
        )
        print("✓ Environment created successfully")
        
        # 测试重置
        print("\n2. Testing environment reset...")
        uav_obs, global_obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  UAV observations shape: {len(uav_obs)} UAVs, each with {len(uav_obs[0])} features")
        print(f"  Global observation shape: {global_obs.shape}")
        
        # 运行几个简单步骤
        print("\n3. Running a few steps...")
        for step in range(3):
            # 生成随机动作
            actions = []
            for i in range(env.num_uavs):
                action = np.random.uniform(-0.5, 0.5, 2)
                actions.append(action)
            
            print(f"\n  Step {step + 1}:")
            print(f"  Actions: {actions}")
            
            # 执行一步
            uav_obs, global_obs, rewards, done, info = env.step(actions)
            
            print(f"  Rewards: {rewards}")
            print(f"  Done: {done}")
            
            if done:
                print("  Environment ended early")
                break
        
        # 显示环境信息
        print("\n4. Environment info:")
        env_info = env.get_environment_info()
        print(f"  Number of UAVs: {env_info['num_uavs']}")
        print(f"  Number of surveillance tasks: {env_info['num_surveillance_tasks']}")
        print(f"  Current emergency tasks: {env_info['current_emergency_tasks']}")
        print(f"  Valid task handling index: {env_info['valid_task_handling_index']:.3f}")
        
        # 显示统计
        print("\n5. Statistics:")
        stats = env_info['stats']
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试渲染（可选，注释掉以避免图形界面问题）
        print("\n6. Testing rendering (will not display if no GUI)...")
        try:
            # 尝试非交互式渲染
            env.render(mode='human')
            print("  Rendering attempted")
            env.close()
        except Exception as e:
            print(f"  Rendering not available: {e}")
        
        print("\n" + "=" * 50)
        print("UAV Environment Test Completed Successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error occurred during testing:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()
        sys.exit(1)