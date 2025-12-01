"""
任务生成器
基于论文中的任务模型生成监控任务和紧急任务
支持真实世界数据集（如旧金山、成都出租车轨迹）的热点分析
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import json
from scipy import stats
from scipy.spatial import KDTree
import time


class TaskGenerationMode(Enum):
    """任务生成模式"""
    RANDOM = "random"              # 随机生成
    UNIFORM = "uniform"            # 均匀分布
    CLUSTERED = "clustered"        # 聚类分布
    REAL_WORLD = "real_world"      # 基于真实数据


@dataclass
class TaskConfig:
    """任务配置"""
    # 监控任务配置
    surveillance_count: int = 50
    surveillance_aoi_threshold_min: int = 25
    surveillance_aoi_threshold_max: int = 40
    surveillance_data_size_min: float = 5.0
    surveillance_data_size_max: float = 15.0
    
    # 紧急任务配置
    emergency_count_min: int = 1
    emergency_count_max: int = 5
    emergency_aoi_threshold_min: int = 10
    emergency_aoi_threshold_max: int = 25
    emergency_data_size_min: float = 30.0
    emergency_data_size_max: float = 70.0
    emergency_area_size_min: float = 50.0
    emergency_area_size_max: float = 150.0
    
    # 任务生成参数
    emergency_probability: float = 0.1
    emergency_interval: int = 6
    emergency_spawn_rate: float = 0.05  # 每个时隙生成概率
    min_distance_between_tasks: float = 50.0  # 任务间最小距离
    
    # 分布参数（用于聚类模式）
    cluster_centers_min: int = 3
    cluster_centers_max: int = 8
    cluster_std_dev: float = 0.15  # 聚类标准差（相对于地图尺寸）


class HotspotDetector:
    """热点检测器 - 从真实数据中检测热点区域"""
    
    def __init__(self, map_size: Tuple[float, float] = (1000, 1000)):
        """
        初始化热点检测器
        
        Args:
            map_size: 地图大小 (宽, 高)
        """
        self.map_width, self.map_height = map_size
        self.points = []
        
    def load_trajectory_data(self, filepath: str, max_points: int = 1000) -> bool:
        """
        加载轨迹数据
        
        Args:
            filepath: 数据文件路径
            max_points: 最大点数
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 支持多种格式
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                # 假设数据包含经纬度或x,y坐标
                if 'longitude' in df.columns and 'latitude' in df.columns:
                    # 简化：直接将经纬度作为坐标
                    self.points = df[['longitude', 'latitude']].values[:max_points]
                elif 'x' in df.columns and 'y' in df.columns:
                    self.points = df[['x', 'y']].values[:max_points]
                else:
                    # 如果没有标准列名，使用前两列
                    self.points = df.iloc[:, :2].values[:max_points]
                    
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # 简化：假设数据是点的列表
                    if isinstance(data, list):
                        self.points = np.array(data)[:max_points]
                        
            elif filepath.endswith('.npy'):
                self.points = np.load(filepath)[:max_points]
                
            else:
                warnings.warn(f"不支持的文件格式: {filepath}")
                return False
                
            # 归一化到地图范围
            if len(self.points) > 0:
                self._normalize_points()
                
            print(f"加载了 {len(self.points)} 个轨迹点")
            return True
            
        except Exception as e:
            warnings.warn(f"加载数据失败: {e}")
            return False
            
    def _normalize_points(self):
        """归一化点到地图范围"""
        if len(self.points) == 0:
            return
            
        # 获取当前范围
        points_array = np.array(self.points)
        min_x, min_y = points_array.min(axis=0)
        max_x, max_y = points_array.max(axis=0)
        
        # 归一化到0-1范围
        points_array[:, 0] = (points_array[:, 0] - min_x) / (max_x - min_x + 1e-10)
        points_array[:, 1] = (points_array[:, 1] - min_y) / (max_y - min_y + 1e-10)
        
        # 缩放到地图尺寸
        points_array[:, 0] *= self.map_width
        points_array[:, 1] *= self.map_height
        
        self.points = points_array
        
    def detect_hotspots(self, num_hotspots: int, method: str = 'kmeans') -> List[np.ndarray]:
        """
        检测热点
        
        Args:
            num_hotspots: 热点数量
            method: 检测方法 ('kmeans', 'dbscan', 'grid', 'top_k')
            
        Returns:
            热点位置列表
        """
        if len(self.points) == 0:
            warnings.warn("没有轨迹数据，返回随机热点")
            return self._generate_random_points(num_hotspots)
            
        if method == 'top_k':
            return self._top_k_density(num_hotspots)
        elif method == 'grid':
            return self._grid_based(num_hotspots)
        elif method == 'kmeans':
            return self._kmeans_clustering(num_hotspots)
        else:
            warnings.warn(f"未知方法: {method}，使用top_k")
            return self._top_k_density(num_hotspots)
            
    def _generate_random_points(self, num_points: int) -> List[np.ndarray]:
        """生成随机点"""
        points = []
        for _ in range(num_points):
            x = np.random.uniform(0, self.map_width)
            y = np.random.uniform(0, self.map_height)
            points.append(np.array([x, y]))
        return points
        
    def _top_k_density(self, k: int, bandwidth: float = 50.0) -> List[np.ndarray]:
        """基于核密度估计的top-k热点检测"""
        if len(self.points) < k:
            return self._generate_random_points(k)
            
        # 使用简单的网格统计代替KDE（避免scipy依赖）
        grid_size = int(self.map_width / 50)  # 50米网格
        density_map = np.zeros((grid_size, grid_size))
        
        for point in self.points:
            i = int(point[0] / self.map_width * grid_size)
            j = int(point[1] / self.map_height * grid_size)
            i = np.clip(i, 0, grid_size-1)
            j = np.clip(j, 0, grid_size-1)
            density_map[i, j] += 1
            
        # 找到最密集的k个网格
        hotspots = []
        for _ in range(k):
            idx = np.argmax(density_map)
            i, j = np.unravel_index(idx, density_map.shape)
            
            # 网格中心作为热点
            x = (i + 0.5) * self.map_width / grid_size
            y = (j + 0.5) * self.map_height / grid_size
            hotspots.append(np.array([x, y]))
            
            # 清除周围区域避免重复
            radius = 2  # 清除2个网格半径
            i_min = max(0, i - radius)
            i_max = min(grid_size, i + radius + 1)
            j_min = max(0, j - radius)
            j_max = min(grid_size, j + radius + 1)
            density_map[i_min:i_max, j_min:j_max] = 0
            
        return hotspots
        
    def _grid_based(self, k: int) -> List[np.ndarray]:
        """基于网格的密度检测"""
        grid_size = int(np.sqrt(k * 4))  # 确保有足够网格
        
        # 创建网格
        x_edges = np.linspace(0, self.map_width, grid_size + 1)
        y_edges = np.linspace(0, self.map_height, grid_size + 1)
        
        # 统计每个网格的点数
        hist, x_edges, y_edges = np.histogram2d(
            self.points[:, 0], self.points[:, 1],
            bins=[x_edges, y_edges]
        )
        
        # 找到最密集的k个网格
        hotspots = []
        flat_indices = np.argsort(hist.flatten())[::-1][:k]
        
        for idx in flat_indices:
            i, j = np.unravel_index(idx, hist.shape)
            x = (x_edges[i] + x_edges[i+1]) / 2
            y = (y_edges[j] + y_edges[j+1]) / 2
            hotspots.append(np.array([x, y]))
            
        return hotspots
        
    def _kmeans_clustering(self, k: int, max_iterations: int = 100) -> List[np.ndarray]:
        """简单的K-means聚类（避免sklearn依赖）"""
        if len(self.points) < k:
            return self._generate_random_points(k)
            
        # 随机初始化聚类中心
        indices = np.random.choice(len(self.points), k, replace=False)
        centroids = self.points[indices].copy()
        
        for _ in range(max_iterations):
            # 分配点到最近的聚类中心
            distances = np.zeros((len(self.points), k))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(self.points - centroid, axis=1)
                
            labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centroids = []
            for i in range(k):
                cluster_points = self.points[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(centroids[i])  # 保持原中心
                    
            new_centroids = np.array(new_centroids)
            
            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return [centroid for centroid in centroids]


class TaskGenerator:
    """任务生成器"""
    
    def __init__(self, 
                 map_size: Tuple[float, float] = (1000, 1000),
                 config: Optional[TaskConfig] = None,
                 mode: TaskGenerationMode = TaskGenerationMode.RANDOM,
                 real_world_data_path: Optional[str] = None):
        """
        初始化任务生成器
        
        Args:
            map_size: 地图大小 (宽, 高) (m)
            config: 任务配置
            mode: 生成模式
            real_world_data_path: 真实世界数据路径
        """
        self.map_width, self.map_height = map_size
        self.config = config or TaskConfig()
        self.mode = mode
        self.real_world_data_path = real_world_data_path
        
        # 热点检测器（用于真实世界模式）
        self.hotspot_detector = HotspotDetector(map_size) if real_world_data_path else None
        
        # 任务ID计数器
        self.next_task_id = 0
        
        # 预生成的监控任务位置
        self.surveillance_locations = []
        
        # 预生成的聚类中心（用于聚类模式）
        self.cluster_centers = []
        
        # 初始化生成位置
        self._initialize_locations()
        
    def _initialize_locations(self):
        """初始化任务位置"""
        # 根据模式生成监控任务位置
        if self.mode == TaskGenerationMode.REAL_WORLD and self.real_world_data_path:
            self._generate_real_world_locations()
        elif self.mode == TaskGenerationMode.CLUSTERED:
            self._generate_clustered_locations()
        elif self.mode == TaskGenerationMode.UNIFORM:
            self._generate_uniform_locations()
        else:  # RANDOM
            self._generate_random_locations()
            
    def _generate_random_locations(self):
        """生成随机位置"""
        count = self.config.surveillance_count
        min_dist = self.config.min_distance_between_tasks
        
        for _ in range(count):
            location = self._generate_location_with_min_distance(min_dist)
            self.surveillance_locations.append(location)
            
    def _generate_uniform_locations(self):
        """生成均匀分布的位置"""
        count = self.config.surveillance_count
        
        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(count)))
        
        for i in range(count):
            # 计算网格坐标
            row = i // grid_size
            col = i % grid_size
            
            # 添加随机偏移避免完全对齐
            x = (col + 0.5 + np.random.uniform(-0.3, 0.3)) / grid_size * self.map_width
            y = (row + 0.5 + np.random.uniform(-0.3, 0.3)) / grid_size * self.map_height
            
            # 确保在边界内
            x = np.clip(x, 10, self.map_width - 10)
            y = np.clip(y, 10, self.map_height - 10)
            
            self.surveillance_locations.append(np.array([x, y]))
            
    def _generate_clustered_locations(self):
        """生成聚类分布的位置"""
        count = self.config.surveillance_count
        
        # 随机选择聚类中心数量
        num_clusters = np.random.randint(
            self.config.cluster_centers_min,
            self.config.cluster_centers_max + 1
        )
        
        # 生成聚类中心
        self.cluster_centers = []
        for _ in range(num_clusters):
            center = np.array([
                np.random.uniform(0.2, 0.8) * self.map_width,
                np.random.uniform(0.2, 0.8) * self.map_height
            ])
            self.cluster_centers.append(center)
            
        # 为每个聚类分配任务数量
        cluster_weights = np.random.dirichlet(np.ones(num_clusters))
        cluster_counts = (cluster_weights * count).astype(int)
        
        # 调整总数
        while sum(cluster_counts) < count:
            cluster_counts[np.random.randint(num_clusters)] += 1
            
        while sum(cluster_counts) > count:
            idx = np.random.randint(num_clusters)
            if cluster_counts[idx] > 1:
                cluster_counts[idx] -= 1
                
        # 生成每个聚类的点
        std_dev = self.config.cluster_std_dev * min(self.map_width, self.map_height)
        
        for cluster_idx, cluster_count in enumerate(cluster_counts):
            center = self.cluster_centers[cluster_idx]
            
            for _ in range(cluster_count):
                # 从正态分布生成点
                location = center + np.random.normal(0, std_dev, 2)
                
                # 确保在地图内
                location[0] = np.clip(location[0], 10, self.map_width - 10)
                location[1] = np.clip(location[1], 10, self.map_height - 10)
                
                self.surveillance_locations.append(location)
                
    def _generate_real_world_locations(self):
        """基于真实世界数据生成位置"""
        if not self.hotspot_detector:
            warnings.warn("没有热点检测器，使用随机模式")
            self._generate_random_locations()
            return
            
        # 加载数据
        success = self.hotspot_detector.load_trajectory_data(self.real_world_data_path)
        if not success:
            warnings.warn("加载数据失败，使用随机模式")
            self._generate_random_locations()
            return
            
        # 检测热点
        hotspots = self.hotspot_detector.detect_hotspots(
            self.config.surveillance_count,
            method='top_k'
        )
        
        # 如果热点不足，补充随机点
        if len(hotspots) < self.config.surveillance_count:
            warnings.warn(f"只检测到 {len(hotspots)} 个热点，补充随机点")
            additional_count = self.config.surveillance_count - len(hotspots)
            additional_points = self.hotspot_detector._generate_random_points(additional_count)
            hotspots.extend(additional_points)
            
        self.surveillance_locations = hotspots[:self.config.surveillance_count]
        
    def _generate_location_with_min_distance(self, min_distance: float) -> np.ndarray:
        """生成满足最小距离要求的位置"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # 随机位置
            location = np.array([
                np.random.uniform(0, self.map_width),
                np.random.uniform(0, self.map_height)
            ])
            
            # 检查距离
            too_close = False
            for existing in self.surveillance_locations:
                if np.linalg.norm(location - existing) < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                return location
                
        # 如果找不到合适位置，返回随机位置
        warnings.warn(f"无法找到满足最小距离 {min_distance} 的位置")
        return np.array([
            np.random.uniform(0, self.map_width),
            np.random.uniform(0, self.map_height)
        ])
        
    def generate_surveillance_task(self, task_id: int, location: np.ndarray) -> Dict[str, Any]:
        """
        生成监控任务
        
        Args:
            task_id: 任务ID
            location: 任务位置
            
        Returns:
            任务字典
        """
        return {
            'task_id': task_id,
            'task_type': 'surveillance',
            'location': location,
            'aoi_threshold': np.random.randint(
                self.config.surveillance_aoi_threshold_min,
                self.config.surveillance_aoi_threshold_max + 1
            ),
            'data_size': np.random.uniform(
                self.config.surveillance_data_size_min,
                self.config.surveillance_data_size_max
            ),
            'data_generation_rate': 1.0,  # MB/timeslot
            'priority': 1.0
        }
        
    def generate_emergency_task(self, task_id: int) -> Dict[str, Any]:
        """
        生成紧急任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务字典
        """
        # 随机位置（避免与现有任务太近）
        location = self._generate_location_with_min_distance(self.config.min_distance_between_tasks)
        
        return {
            'task_id': task_id,
            'task_type': 'emergency',
            'location': location,
            'aoi_threshold': np.random.randint(
                self.config.emergency_aoi_threshold_min,
                self.config.emergency_aoi_threshold_max + 1
            ),
            'data_size': np.random.uniform(
                self.config.emergency_data_size_min,
                self.config.emergency_data_size_max
            ),
            'area_size': np.random.uniform(
                self.config.emergency_area_size_min,
                self.config.emergency_area_size_max
            ),
            'images_needed': np.random.randint(5, 15),
            'priority': 2.0
        }
        
    def generate_initial_surveillance_tasks(self) -> List[Dict[str, Any]]:
        """生成初始监控任务"""
        tasks = []
        
        for i, location in enumerate(self.surveillance_locations):
            task = self.generate_surveillance_task(self.next_task_id, location)
            tasks.append(task)
            self.next_task_id += 1
            
        return tasks
        
    def generate_initial_emergency_tasks(self) -> List[Dict[str, Any]]:
        """生成初始紧急任务"""
        tasks = []
        
        num_emergencies = np.random.randint(
            self.config.emergency_count_min,
            self.config.emergency_count_max + 1
        )
        
        for _ in range(num_emergencies):
            task = self.generate_emergency_task(self.next_task_id)
            tasks.append(task)
            self.next_task_id += 1
            
        return tasks
        
    def should_generate_emergency(self, current_time_slot: int) -> bool:
        """
        检查是否应该生成紧急任务
        
        Args:
            current_time_slot: 当前时隙
            
        Returns:
            bool: 是否生成
        """
        # 基于固定间隔
        if current_time_slot % self.config.emergency_interval == 0:
            return True
            
        # 基于概率
        if np.random.random() < self.config.emergency_spawn_rate:
            return True
            
        return False
        
    def generate_dynamic_emergency_task(self) -> Optional[Dict[str, Any]]:
        """
        生成动态紧急任务
        
        Returns:
            紧急任务字典，如果没有生成则返回None
        """
        if self.should_generate_emergency(np.random.randint(0, 100)):  # 使用随机时间模拟
            task = self.generate_emergency_task(self.next_task_id)
            self.next_task_id += 1
            return task
            
        return None
        
    def get_surveillance_locations(self) -> List[np.ndarray]:
        """获取监控任务位置"""
        return self.surveillance_locations.copy()
        
    def get_cluster_centers(self) -> List[np.ndarray]:
        """获取聚类中心（仅聚类模式有效）"""
        return self.cluster_centers.copy()
        
    def visualize_distribution(self, save_path: Optional[str] = None):
        """可视化任务分布"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 设置坐标轴
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Task Distribution - Mode: {self.mode.value}')
        
        # 绘制监控任务点
        if self.surveillance_locations:
            locations = np.array(self.surveillance_locations)
            ax.scatter(locations[:, 0], locations[:, 1], 
                      c='green', s=50, alpha=0.6, label='Surveillance Tasks')
                      
        # 绘制聚类中心（如果存在）
        if self.cluster_centers:
            centers = np.array(self.cluster_centers)
            ax.scatter(centers[:, 0], centers[:, 1], 
                      c='red', s=200, marker='*', label='Cluster Centers')
                      
            # 绘制聚类区域
            std_dev = self.config.cluster_std_dev * min(self.map_width, self.map_height)
            for center in centers:
                circle = plt.Circle(center, 2*std_dev, 
                                   fill=False, linestyle='--', alpha=0.3, color='red')
                ax.add_patch(circle)
                
        # 绘制地图边界
        rect = patches.Rectangle((0, 0), self.map_width, self.map_height,
                                linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"分布图已保存到: {save_path}")
            
        plt.show()
        
    def generate_traffic_heatmap(self, resolution: int = 50) -> np.ndarray:
        """
        生成交通流量热图
        
        Args:
            resolution: 热图分辨率
            
        Returns:
            热图矩阵
        """
        heatmap = np.zeros((resolution, resolution))
        
        # 为每个位置分配"流量"
        for location in self.surveillance_locations:
            i = int(location[0] / self.map_width * resolution)
            j = int(location[1] / self.map_height * resolution)
            i = np.clip(i, 0, resolution-1)
            j = np.clip(j, 0, resolution-1)
            
            # 增加流量值（模拟热点区域）
            heatmap[i, j] += 1
            
        # 对流量进行高斯模糊（模拟影响范围）
        from scipy.ndimage import gaussian_filter
        if len(self.surveillance_locations) > 0:
            heatmap = gaussian_filter(heatmap, sigma=1.0)
            
        return heatmap
        
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'mode': self.mode.value,
            'map_size': (self.map_width, self.map_height),
            'surveillance_count': len(self.surveillance_locations),
            'config': {
                'surveillance_aoi_threshold_range': (
                    self.config.surveillance_aoi_threshold_min,
                    self.config.surveillance_aoi_threshold_max
                ),
                'emergency_aoi_threshold_range': (
                    self.config.emergency_aoi_threshold_min,
                    self.config.emergency_aoi_threshold_max
                ),
                'emergency_interval': self.config.emergency_interval,
                'emergency_spawn_rate': self.config.emergency_spawn_rate
            }
        }


# 高级任务生成器（支持更复杂的模式）
class AdvancedTaskGenerator(TaskGenerator):
    """高级任务生成器 - 支持更复杂的任务模式"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 时间相关的任务生成
        self.time_patterns = {
            'morning': {'surveillance_multiplier': 1.2, 'emergency_probability': 0.08},
            'afternoon': {'surveillance_multiplier': 1.0, 'emergency_probability': 0.1},
            'evening': {'surveillance_multiplier': 1.5, 'emergency_probability': 0.15},
            'night': {'surveillance_multiplier': 0.7, 'emergency_probability': 0.05}
        }
        
        # 区域相关的重要性
        self.region_importance = {}  # 可动态设置
        
    def set_region_importance(self, region_bounds: Tuple[float, float, float, float], 
                             importance: float):
        """
        设置区域重要性
        
        Args:
            region_bounds: (x_min, y_min, x_max, y_max) 区域边界
            importance: 重要性因子 (0.0-2.0)
        """
        self.region_importance[region_bounds] = importance
        
    def generate_task_with_importance(self, task_type: str, 
                                     current_time_of_day: str = 'afternoon') -> Dict[str, Any]:
        """
        基于重要性和时间生成任务
        
        Args:
            task_type: 'surveillance' 或 'emergency'
            current_time_of_day: 当前时间段
            
        Returns:
            任务字典
        """
        if task_type == 'surveillance':
            task = self.generate_surveillance_task(self.next_task_id, 
                                                  self.surveillance_locations[0])
        else:
            task = self.generate_emergency_task(self.next_task_id)
            
        self.next_task_id += 1
        
        # 根据时间调整参数
        time_pattern = self.time_patterns.get(current_time_of_day, 
                                             self.time_patterns['afternoon'])
        
        if task_type == 'surveillance':
            task['data_size'] *= time_pattern['surveillance_multiplier']
        else:
            task['priority'] *= 1.0 + (1.0 - time_pattern['emergency_probability'])
            
        # 根据区域重要性调整
        task_location = task['location']
        for region_bounds, importance in self.region_importance.items():
            x_min, y_min, x_max, y_max = region_bounds
            if (x_min <= task_location[0] <= x_max and 
                y_min <= task_location[1] <= y_max):
                task['priority'] *= importance
                task['aoi_threshold'] = max(1, int(task['aoi_threshold'] / importance))
                break
                
        return task
        
    def generate_correlated_emergencies(self, num_correlated: int = 2, 
                                       correlation_radius: float = 200.0) -> List[Dict[str, Any]]:
        """
        生成相关紧急任务（如连环事故）
        
        Args:
            num_correlated: 相关任务数量
            correlation_radius: 相关半径
            
        Returns:
            相关任务列表
        """
        # 生成第一个任务作为中心
        center_task = self.generate_emergency_task(self.next_task_id)
        self.next_task_id += 1
        center_location = center_task['location']
        
        tasks = [center_task]
        
        # 生成相关任务
        for i in range(1, num_correlated):
            # 在中心附近生成任务
            angle = 2 * np.pi * i / num_correlated
            distance = np.random.uniform(0.3, 1.0) * correlation_radius
            
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            location = center_location + np.array([dx, dy])
            
            # 确保在地图内
            location[0] = np.clip(location[0], 10, self.map_width - 10)
            location[1] = np.clip(location[1], 10, self.map_height - 10)
            
            # 生成相关任务
            task = self.generate_emergency_task(self.next_task_id)
            task['location'] = location
            task['aoi_threshold'] = center_task['aoi_threshold']  # 相同紧急程度
            task['priority'] = center_task['priority'] * 0.8  # 稍低优先级
            
            tasks.append(task)
            self.next_task_id += 1
            
        return tasks


# 测试函数
def test_task_generator():
    """测试任务生成器"""
    print("=" * 60)
    print("测试任务生成器")
    print("=" * 60)
    
    try:
        # 测试不同模式
        modes = [
            (TaskGenerationMode.RANDOM, "随机模式"),
            (TaskGenerationMode.UNIFORM, "均匀模式"),
            (TaskGenerationMode.CLUSTERED, "聚类模式")
        ]
        
        for mode, name in modes:
            print(f"\n测试: {name}")
            print("-" * 40)
            
            # 创建生成器
            generator = TaskGenerator(
                map_size=(800, 600),  # 使用更小的地图以加快测试
                config=TaskConfig(surveillance_count=10),  # 减少任务数量
                mode=mode
            )
            
            # 生成监控任务
            surveillance_tasks = generator.generate_initial_surveillance_tasks()
            print(f"生成监控任务: {len(surveillance_tasks)} 个")
            
            # 生成紧急任务
            emergency_tasks = generator.generate_initial_emergency_tasks()
            print(f"生成紧急任务: {len(emergency_tasks)} 个")
            
            # 显示示例任务
            if surveillance_tasks:
                task = surveillance_tasks[0]
                print(f"监控任务示例:")
                print(f"  位置: {task['location']}")
                print(f"  AoI阈值: {task['aoi_threshold']}")
                print(f"  数据大小: {task['data_size']:.1f} MB")
                
            if emergency_tasks:
                task = emergency_tasks[0]
                print(f"紧急任务示例:")
                print(f"  位置: {task['location']}")
                print(f"  AoI阈值: {task['aoi_threshold']}")
                print(f"  区域大小: {task['area_size']:.1f} m²")
                print(f"  所需图像: {task['images_needed']}")
                
            # 测试动态生成
            print(f"\n测试动态生成:")
            for i in range(3):  # 减少测试次数
                emergency = generator.generate_dynamic_emergency_task()
                if emergency:
                    print(f"  时隙 {i}: 生成紧急任务 ID={emergency['task_id']}")
                else:
                    print(f"  时隙 {i}: 未生成任务")
                    
        # 测试高级生成器
        print(f"\n\n测试高级任务生成器")
        print("-" * 40)
        
        advanced_gen = AdvancedTaskGenerator(
            map_size=(600, 600),
            config=TaskConfig(surveillance_count=8),
            mode=TaskGenerationMode.CLUSTERED
        )
        
        # 设置区域重要性
        advanced_gen.set_region_importance((100, 100, 300, 300), 1.5)
        advanced_gen.set_region_importance((400, 400, 600, 600), 0.7)
        
        # 生成基于时间的任务
        print("生成基于时间的任务:")
        for time_of_day in ['morning', 'afternoon']:  # 只测试两个时间段
            task = advanced_gen.generate_task_with_importance('surveillance', time_of_day)
            print(f"  {time_of_day}: 数据大小={task['data_size']:.1f} MB, 优先级={task['priority']:.1f}")
            
        # 生成相关紧急任务
        print("\n生成相关紧急任务:")
        correlated = advanced_gen.generate_correlated_emergencies(2, 100.0)
        print(f"  生成 {len(correlated)} 个相关任务")
        for i, task in enumerate(correlated):
            print(f"    任务 {i}: 位置={task['location']}, 优先级={task['priority']:.1f}")
            
        print(f"\n{'='*60}")
        print("任务生成器测试完成!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# 简单测试函数（最小化测试）
def minimal_test():
    """最小化测试"""
    print("开始最小化测试...")
    
    try:
        # 只测试随机模式
        generator = TaskGenerator(
            map_size=(500, 500),
            config=TaskConfig(surveillance_count=5),
            mode=TaskGenerationMode.RANDOM
        )
        
        print("✓ 任务生成器创建成功")
        
        tasks = generator.generate_initial_surveillance_tasks()
        print(f"✓ 生成 {len(tasks)} 个监控任务")
        
        emergencies = generator.generate_initial_emergency_tasks()
        print(f"✓ 生成 {len(emergencies)} 个紧急任务")
        
        print("\n最小化测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


# 主入口点
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='任务生成器测试')
    parser.add_argument('--test', type=str, choices=['full', 'minimal', 'quick'], 
                       default='quick', help='测试模式')
    parser.add_argument('--mode', type=str, 
                       choices=['random', 'uniform', 'clustered', 'real_world'],
                       default='random', help='生成模式')
    
    args = parser.parse_args()
    
    print("Python版本:", sys.version)
    print("当前目录:", sys.path[0])
    
    if args.test == 'minimal':
        minimal_test()
    elif args.test == 'full':
        test_task_generator()
    else:  # quick test
        print("\n快速测试模式...")
        
        try:
            from enum import Enum
            import numpy as np
            
            print("✓ 基本导入成功")
            
            # 创建简单生成器
            config = TaskConfig(surveillance_count=3)
            generator = TaskGenerator(
                map_size=(400, 400),
                config=config,
                mode=TaskGenerationMode[args.mode.upper()]
            )
            
            print(f"✓ {args.mode} 模式生成器创建成功")
            
            # 生成任务
            surveillance = generator.generate_initial_surveillance_tasks()
            emergencies = generator.generate_initial_emergency_tasks()
            
            print(f"监控任务: {len(surveillance)} 个")
            print(f"紧急任务: {len(emergencies)} 个")
            
            if surveillance:
                task = surveillance[0]
                print(f"\n第一个监控任务:")
                print(f"  ID: {task['task_id']}")
                print(f"  位置: ({task['location'][0]:.1f}, {task['location'][1]:.1f})")
                print(f"  AoI阈值: {task['aoi_threshold']}")
                print(f"  数据大小: {task['data_size']:.1f} MB")
            
            print("\n✓ 快速测试完成!")
            
        except Exception as e:
            print(f"\n✗ 测试错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()