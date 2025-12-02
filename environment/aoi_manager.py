"""
AoI管理器
用于管理所有任务的AoI（信息年龄）
基于论文中的AoI模型：负责监控和紧急任务的AoI更新和统计
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    SURVEILLANCE = "surveillance"  # 监控任务
    EMERGENCY = "emergency"       # 紧急任务


@dataclass
class TaskConfig:
    """任务配置"""
    task_id: int
    task_type: TaskType
    aoi_threshold: int  # AoI阈值
    location: Tuple[float, float]  # 位置坐标
    initial_aoi: int = 1  # 初始AoI
    
    def __post_init__(self):
        """后初始化验证"""
        if self.aoi_threshold <= 0:
            raise ValueError(f"AoI阈值必须为正数: {self.aoi_threshold}")
        if self.initial_aoi < 1:
            raise ValueError(f"初始AoI必须≥1: {self.initial_aoi}")


class AoIManager:
    """AoI管理器"""
    
    def __init__(self, total_timeslots: int = 1000):
        """
        初始化AoI管理器
        
        Args:
            total_timeslots: 总时隙数
        """
        self.total_timeslots = total_timeslots
        self.current_timeslot = 0
        
        # 任务存储
        self.tasks: Dict[int, TaskConfig] = {}  # 任务ID -> 任务配置
        self.task_aoi: Dict[int, List[int]] = {}  # 任务ID -> AoI历史列表
        self.task_valid_flags: Dict[int, List[bool]] = {}  # 任务ID -> 有效标志列表
        self.task_handling_times: Dict[int, List[int]] = {}  # 任务ID -> 处理时间列表
        
        # 统计信息
        self.stats = {
            "total_valid_slots": 0,
            "total_invalid_slots": 0,
            "total_handlings": 0,
            "task_type_counts": {
                TaskType.SURVEILLANCE: 0,
                TaskType.EMERGENCY: 0
            }
        }
        
        # 性能指标缓存
        self.valid_handling_ratios = {}
        self.valid_task_handling_index_cache = None
        
    def add_task(self, task_config: TaskConfig) -> None:
        """
        添加新任务
        
        Args:
            task_config: 任务配置
        """
        task_id = task_config.task_id
        
        if task_id in self.tasks:
            raise ValueError(f"任务ID已存在: {task_id}")
        
        # 添加任务
        self.tasks[task_id] = task_config
        self.task_aoi[task_id] = [task_config.initial_aoi]
        self.task_valid_flags[task_id] = [self._is_valid(task_config.initial_aoi, task_config.aoi_threshold)]
        self.task_handling_times[task_id] = []
        
        # 更新统计
        self.stats["task_type_counts"][task_config.task_type] += 1
        
        # 更新当前时隙的有效标志（如果需要填充历史）
        while len(self.task_aoi[task_id]) < self.current_timeslot + 1:
            self.task_aoi[task_id].append(task_config.initial_aoi)
            self.task_valid_flags[task_id].append(
                self._is_valid(task_config.initial_aoi, task_config.aoi_threshold)
            )
    
    def add_surveillance_task(self, task_id: int, location: Tuple[float, float], 
                            aoi_threshold: int = 35) -> None:
        """
        添加监控任务
        
        Args:
            task_id: 任务ID
            location: 位置坐标
            aoi_threshold: AoI阈值
        """
        config = TaskConfig(
            task_id=task_id,
            task_type=TaskType.SURVEILLANCE,
            aoi_threshold=aoi_threshold,
            location=location,
            initial_aoi=1
        )
        self.add_task(config)
    
    def add_emergency_task(self, task_id: int, location: Tuple[float, float],
                          aoi_threshold: int = 20, spawn_timeslot: int = None) -> None:
        """
        添加紧急任务
        
        Args:
            task_id: 任务ID
            location: 位置坐标
            aoi_threshold: AoI阈值
            spawn_timeslot: 任务出现的时隙（如果为None，则在当前时隙出现）
        """
        if spawn_timeslot is None:
            spawn_timeslot = self.current_timeslot
            
        # 如果需要，将当前时隙调整到任务出现之前
        original_timeslot = self.current_timeslot
        self.current_timeslot = spawn_timeslot
        
        config = TaskConfig(
            task_id=task_id,
            task_type=TaskType.EMERGENCY,
            aoi_threshold=aoi_threshold,
            location=location,
            initial_aoi=1
        )
        self.add_task(config)
        
        # 恢复原始时隙
        self.current_timeslot = original_timeslot
    
    def update_timeslot(self, handled_tasks: Optional[Set[int]] = None) -> None:
        """
        更新到下一个时隙
        
        Args:
            handled_tasks: 在当前时隙被处理的任务ID集合
        """
        if handled_tasks is None:
            handled_tasks = set()
        
        self.current_timeslot += 1
        
        # 更新所有任务的AoI
        for task_id, task_config in self.tasks.items():
            # 获取上一个时隙的AoI
            if len(self.task_aoi[task_id]) > self.current_timeslot - 1:
                prev_aoi = self.task_aoi[task_id][-1]
            else:
                # 填充缺失的历史AoI
                while len(self.task_aoi[task_id]) < self.current_timeslot:
                    self.task_aoi[task_id].append(task_config.initial_aoi)
                    self.task_valid_flags[task_id].append(
                        self._is_valid(task_config.initial_aoi, task_config.aoi_threshold)
                    )
                prev_aoi = task_config.initial_aoi
            
            # 计算当前时隙的AoI
            if task_id in handled_tasks:
                current_aoi = 1  # 任务被处理，重置为1
                self.task_handling_times[task_id].append(self.current_timeslot)
                self.stats["total_handlings"] += 1
            else:
                current_aoi = prev_aoi + 1  # 任务未被处理，AoI加1
            
            # 存储当前时隙的AoI
            self.task_aoi[task_id].append(current_aoi)
            
            # 检查是否有效
            is_valid = self._is_valid(current_aoi, task_config.aoi_threshold)
            self.task_valid_flags[task_id].append(is_valid)
            
            # 更新统计
            if is_valid:
                self.stats["total_valid_slots"] += 1
            else:
                self.stats["total_invalid_slots"] += 1
    
    def get_task_aoi(self, task_id: int, timeslot: Optional[int] = None) -> int:
        """
        获取任务的AoI
        
        Args:
            task_id: 任务ID
            timeslot: 时隙（如果为None，则使用当前时隙）
            
        Returns:
            任务的AoI
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        if timeslot is None:
            timeslot = self.current_timeslot
        
        if timeslot < 0 or timeslot > self.current_timeslot:
            raise ValueError(f"时隙超出范围: {timeslot}")
        
        # 确保有该时隙的数据
        if timeslot >= len(self.task_aoi[task_id]):
            return self.tasks[task_id].initial_aoi
        
        return self.task_aoi[task_id][timeslot]
    
    def is_task_valid(self, task_id: int, timeslot: Optional[int] = None) -> bool:
        """
        检查任务在当前时隙是否有效
        
        Args:
            task_id: 任务ID
            timeslot: 时隙（如果为None，则使用当前时隙）
            
        Returns:
            任务是否有效（AoI ≤ 阈值）
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        if timeslot is None:
            timeslot = self.current_timeslot
        
        if timeslot < 0 or timeslot > self.current_timeslot:
            return False
        
        # 确保有该时隙的数据
        if timeslot >= len(self.task_valid_flags[task_id]):
            aoi = self.get_task_aoi(task_id, timeslot)
            return self._is_valid(aoi, self.tasks[task_id].aoi_threshold)
        
        return self.task_valid_flags[task_id][timeslot]
    
    def _is_valid(self, aoi: int, threshold: int) -> bool:
        """
        检查AoI是否有效
        
        Args:
            aoi: 当前AoI
            threshold: AoI阈值
            
        Returns:
            AoI是否有效
        """
        return aoi <= threshold
    
    def get_valid_handling_ratio(self, task_type: TaskType) -> float:
        """
        计算任务类型的有效处理比例（公式6）
        
        Args:
            task_type: 任务类型
            
        Returns:
            有效处理比例 I_m ∈ [0, 1]
        """
        if task_type in self.valid_handling_ratios:
            return self.valid_handling_ratios[task_type]
        
        # 获取该类型的所有任务
        task_ids = [
            task_id for task_id, config in self.tasks.items()
            if config.task_type == task_type
        ]
        
        if not task_ids:
            return 0.0
        
        total_valid_slots = 0
        total_slots = 0
        
        for task_id in task_ids:
            # 只考虑任务存在的时隙（从任务出现到当前时隙）
            task_config = self.tasks[task_id]
            
            # 获取有效标志列表
            valid_flags = self.task_valid_flags[task_id]
            
            # 计算有效时隙数
            task_valid_slots = sum(1 for is_valid in valid_flags if is_valid)
            task_total_slots = len(valid_flags)
            
            total_valid_slots += task_valid_slots
            total_slots += task_total_slots
        
        # 计算比例
        if total_slots == 0:
            ratio = 0.0
        else:
            ratio = total_valid_slots / total_slots
        
        self.valid_handling_ratios[task_type] = ratio
        return ratio
    
    def get_valid_task_handling_index(self, energy_consumption_ratio: float = 1.0) -> float:
        """
        计算有效任务处理指数（公式7）
        
        Args:
            energy_consumption_ratio: 能量消耗比例 η
            
        Returns:
            有效任务处理指数 I
        """
        if energy_consumption_ratio <= 0:
            raise ValueError(f"能量消耗比例必须为正数: {energy_consumption_ratio}")
        
        # 获取所有任务类型的有效处理比例
        task_types = set(config.task_type for config in self.tasks.values())
        
        if not task_types:
            return 0.0
        
        # 计算最小有效处理比例
        min_ratio = min(self.get_valid_handling_ratio(task_type) for task_type in task_types)
        
        # 计算有效任务处理指数
        valid_index = min_ratio / energy_consumption_ratio
        
        self.valid_task_handling_index_cache = {
            "min_ratio": min_ratio,
            "energy_ratio": energy_consumption_ratio,
            "valid_index": valid_index
        }
        
        return valid_index
    
    def get_task_type_stats(self, task_type: TaskType) -> Dict[str, Any]:
        """
        获取任务类型的统计信息
        
        Args:
            task_type: 任务类型
            
        Returns:
            统计信息字典
        """
        task_ids = [
            task_id for task_id, config in self.tasks.items()
            if config.task_type == task_type
        ]
        
        if not task_ids:
            return {
                "count": 0,
                "avg_aoi": 0.0,
                "valid_ratio": 0.0,
                "avg_handling_interval": 0.0
            }
        
        # 计算平均AoI
        total_aoi = 0
        total_slots = 0
        
        # 计算处理间隔
        handling_intervals = []
        
        for task_id in task_ids:
            aoi_history = self.task_aoi[task_id]
            total_aoi += sum(aoi_history)
            total_slots += len(aoi_history)
            
            # 分析处理间隔
            handling_times = self.task_handling_times[task_id]
            if len(handling_times) > 1:
                intervals = np.diff(sorted(handling_times))
                handling_intervals.extend(intervals.tolist())
        
        avg_aoi = total_aoi / total_slots if total_slots > 0 else 0.0
        valid_ratio = self.get_valid_handling_ratio(task_type)
        avg_interval = np.mean(handling_intervals) if handling_intervals else 0.0
        
        return {
            "count": len(task_ids),
            "avg_aoi": avg_aoi,
            "valid_ratio": valid_ratio,
            "avg_handling_interval": avg_interval,
            "handling_intervals": handling_intervals
        }
    
    def get_aoi_heatmap(self, grid_resolution: int = 10) -> np.ndarray:
        """
        获取AoI热力图
        
        Args:
            grid_resolution: 网格分辨率
            
        Returns:
            AoI热力图矩阵
        """
        # 获取所有任务位置
        locations = np.array([config.location for config in self.tasks.values()])
        
        if len(locations) == 0:
            return np.zeros((grid_resolution, grid_resolution))
        
        # 创建网格
        x_min, y_min = locations.min(axis=0)
        x_max, y_max = locations.max(axis=0)
        
        # 扩展边界
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # 创建网格点
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        
        # 初始化热力图
        heatmap = np.zeros((grid_resolution, grid_resolution))
        
        # 计算每个网格点的平均AoI
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                grid_x = x_grid[i]
                grid_y = y_grid[j]
                
                # 计算到所有任务的距离
                distances = np.sqrt((locations[:, 0] - grid_x)**2 + (locations[:, 1] - grid_y)**2)
                
                # 找到最近的任务
                if len(distances) > 0:
                    min_dist_idx = np.argmin(distances)
                    task_id = list(self.tasks.keys())[min_dist_idx]
                    heatmap[i, j] = self.get_task_aoi(task_id)
        
        return heatmap
    
    def reset(self) -> None:
        """重置AoI管理器"""
        self.current_timeslot = 0
        self.tasks.clear()
        self.task_aoi.clear()
        self.task_valid_flags.clear()
        self.task_handling_times.clear()
        
        # 重置统计
        self.stats = {
            "total_valid_slots": 0,
            "total_invalid_slots": 0,
            "total_handlings": 0,
            "task_type_counts": {
                TaskType.SURVEILLANCE: 0,
                TaskType.EMERGENCY: 0
            }
        }
        
        # 清除缓存
        self.valid_handling_ratios.clear()
        self.valid_task_handling_index_cache = None
    
    def print_summary(self) -> None:
        """打印摘要信息"""
        print("=" * 60)
        print("AoI管理器摘要")
        print("=" * 60)
        
        print(f"当前时隙: {self.current_timeslot}")
        print(f"总任务数: {len(self.tasks)}")
        print(f"监控任务数: {self.stats['task_type_counts'][TaskType.SURVEILLANCE]}")
        print(f"紧急任务数: {self.stats['task_type_counts'][TaskType.EMERGENCY]}")
        print()
        
        # 打印任务类型统计
        for task_type in [TaskType.SURVEILLANCE, TaskType.EMERGENCY]:
            stats = self.get_task_type_stats(task_type)
            if stats["count"] > 0:
                print(f"{task_type.value}任务统计:")
                print(f"  任务数量: {stats['count']}")
                print(f"  平均AoI: {stats['avg_aoi']:.2f}")
                print(f"  有效处理比例: {stats['valid_ratio']:.3f}")
                if stats['avg_handling_interval'] > 0:
                    print(f"  平均处理间隔: {stats['avg_handling_interval']:.2f} 时隙")
                print()
        
        # 计算整体统计
        total_slots = self.stats["total_valid_slots"] + self.stats["total_invalid_slots"]
        if total_slots > 0:
            overall_valid_ratio = self.stats["total_valid_slots"] / total_slots
            print(f"整体统计:")
            print(f"  总有效时隙: {self.stats['total_valid_slots']}")
            print(f"  总无效时隙: {self.stats['total_invalid_slots']}")
            print(f"  总处理次数: {self.stats['total_handlings']}")
            print(f"  整体有效比例: {overall_valid_ratio:.3f}")
        
        print("=" * 60)


def test_aoi_manager():
    """测试AoI管理器"""
    print("测试AoI管理器...")
    
    # 创建AoI管理器
    aoi_manager = AoIManager(total_timeslots=100)
    
    # 添加监控任务
    print("\n1. 添加监控任务...")
    aoi_manager.add_surveillance_task(1, (10, 20), aoi_threshold=35)
    aoi_manager.add_surveillance_task(2, (30, 40), aoi_threshold=35)
    aoi_manager.add_surveillance_task(3, (50, 60), aoi_threshold=35)
    
    # 添加紧急任务
    print("\n2. 添加紧急任务...")
    aoi_manager.add_emergency_task(4, (15, 25), aoi_threshold=20)
    aoi_manager.add_emergency_task(5, (35, 45), aoi_threshold=15)
    
    # 模拟时隙更新
    print("\n3. 模拟时隙更新...")
    for t in range(10):
        # 随机处理一些任务
        handled_tasks = set()
        if t % 3 == 0:
            handled_tasks.add(1)  # 每3个时隙处理任务1
        if t % 5 == 0:
            handled_tasks.add(4)  # 每5个时隙处理任务4
        
        aoi_manager.update_timeslot(handled_tasks)
        
        print(f"时隙 {t}: 处理任务 {handled_tasks}")
        
        # 检查任务状态
        for task_id in [1, 4]:
            aoi = aoi_manager.get_task_aoi(task_id)
            is_valid = aoi_manager.is_task_valid(task_id)
            print(f"  任务 {task_id}: AoI={aoi}, 有效={is_valid}")
    
    # 计算性能指标
    print("\n4. 计算性能指标...")
    surveillance_ratio = aoi_manager.get_valid_handling_ratio(TaskType.SURVEILLANCE)
    emergency_ratio = aoi_manager.get_valid_handling_ratio(TaskType.EMERGENCY)
    
    print(f"监控任务有效处理比例: {surveillance_ratio:.3f}")
    print(f"紧急任务有效处理比例: {emergency_ratio:.3f}")
    
    # 计算有效任务处理指数（假设能量消耗比例为0.67）
    valid_index = aoi_manager.get_valid_task_handling_index(energy_consumption_ratio=0.67)
    print(f"有效任务处理指数: {valid_index:.3f}")
    
    # 获取统计信息
    print("\n5. 获取统计信息...")
    surveillance_stats = aoi_manager.get_task_type_stats(TaskType.SURVEILLANCE)
    emergency_stats = aoi_manager.get_task_type_stats(TaskType.EMERGENCY)
    
    print("监控任务统计:")
    for key, value in surveillance_stats.items():
        if key != "handling_intervals":
            print(f"  {key}: {value}")
    
    print("\n紧急任务统计:")
    for key, value in emergency_stats.items():
        if key != "handling_intervals":
            print(f"  {key}: {value}")
    
    # 获取AoI热力图
    print("\n6. 生成AoI热力图...")
    heatmap = aoi_manager.get_aoi_heatmap(grid_resolution=5)
    print(f"热力图形状: {heatmap.shape}")
    print("热力图预览:")
    print(heatmap)
    
    # 打印摘要
    print("\n7. AoI管理器摘要:")
    aoi_manager.print_summary()
    
    print("\nAoI管理器测试完成!")


if __name__ == "__main__":
    test_aoi_manager()