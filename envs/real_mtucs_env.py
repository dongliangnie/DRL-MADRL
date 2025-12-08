import torch
import numpy as np
from typing import List, Tuple, Dict

class RealMTUCSEnvironment:
    """
    真实的多无人机任务分配环境 (Multi-UAV Task Coverage and Scheduling)
    
    特点：
    - 100个动态任务，每个任务有位置、紧急度、剩余时间
    - 5个无人机，有速度、能量、位置等物理属性
    - 真实的物理运动模型
    - 基于距离、紧急度、完成率的奖励函数
    """
    def __init__(
        self, 
        num_uavs: int = 5, 
        num_tasks: int = 100,
        map_size: float = 1000.0,  # 地图尺寸 (米)
        max_steps: int = 1000,
        uav_speed: float = 20.0,   # 无人机速度 (米/秒)
        dt: float = 1.0            # 时间步长 (秒)
    ):
        self.num_uavs = num_uavs
        self.num_tasks = num_tasks
        self.map_size = map_size
        self.max_steps = max_steps
        self.uav_speed = uav_speed
        self.dt = dt
        
        # === 维度定义 (与 train.py 兼容) ===
        # 全局观测: [UAV状态(5*6), 任务状态(100*5), 统计信息(10)]
        self.global_obs_dim = (num_uavs * 6) + (num_tasks * 5) + 10
        
        # 局部观测 (单个UAV): [自身状态(6), 最近5个任务(5*5), 相对信息(10)]
        self.low_obs_dim = 6 + (5 * 5) + 10  # = 41
        
        # 动作: [vx, vy, task_selection, hover_flag]
        self.low_action_dim = 4
        
        # 高层动作空间: 为每个UAV选择一个任务索引 (0~99)
        self.high_action_dim = num_tasks
        
        # === 状态变量 ===
        self.current_step = 0
        self.uav_positions = np.zeros((num_uavs, 2))      # (x, y)
        self.uav_velocities = np.zeros((num_uavs, 2))     # (vx, vy)
        self.uav_energy = np.zeros(num_uavs)              # 剩余能量 [0, 1]
        self.uav_task_assignments = [-1] * num_uavs       # 当前目标任务索引
        
        self.task_positions = np.zeros((num_tasks, 2))    # 任务位置
        self.task_urgency = np.zeros(num_tasks)           # 紧急度 [0, 1]
        self.task_remaining_time = np.zeros(num_tasks)    # 剩余时间 (秒)
        self.task_status = np.zeros(num_tasks)            # 0=待处理, 1=进行中, 2=已完成, -1=超时失败
        self.task_processing_progress = np.zeros(num_tasks)  # 处理进度 [0, 1]
        
        # === 统计信息 ===
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_distance_traveled = 0.0
        self.total_reward = 0.0

    def reset(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """重置环境"""
        self.current_step = 0
        
        # === 初始化 UAV ===
        # 均匀分布在地图边缘作为基地
        angles = np.linspace(0, 2*np.pi, self.num_uavs, endpoint=False)
        radius = self.map_size * 0.3
        self.uav_positions = np.stack([
            self.map_size/2 + radius * np.cos(angles),
            self.map_size/2 + radius * np.sin(angles)
        ], axis=1)
        
        self.uav_velocities = np.zeros((self.num_uavs, 2))
        self.uav_energy = np.ones(self.num_uavs) * 30.0  # ← 3倍能量（从 1.0 改为 3.0）
        self.uav_task_assignments = [-1] * self.num_uavs
        
        # === 初始化任务 (随机分布) ===
        self.task_positions = np.random.rand(self.num_tasks, 2) * self.map_size
        self.task_urgency = np.random.rand(self.num_tasks) * 0.5 + 0.3  # [0.3, 0.8]
        self.task_remaining_time = np.random.rand(self.num_tasks) * 1500 + 1000  # [1000, 2500]秒
        self.task_status = np.zeros(self.num_tasks)
        self.task_processing_progress = np.zeros(self.num_tasks)
        
        # 重置统计
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_distance_traveled = 0.0
        self.total_reward = 0.0
        
        global_obs = self._get_global_obs()
        low_obs_list = self._get_low_obs()
        
        return global_obs, low_obs_list

    def _get_global_obs(self) -> torch.Tensor:
        """构造全局观测 (用于高层策略)"""
        obs_components = []
        
        # 1. UAV 状态 (5 * 6 = 30)
        for i in range(self.num_uavs):
            uav_state = np.concatenate([
                self.uav_positions[i] / self.map_size,
                self.uav_velocities[i] / self.uav_speed,
                [self.uav_energy[i]],
                [float(self.uav_task_assignments[i]) / self.num_tasks]
            ])
            obs_components.append(uav_state)
        
        # 2. 任务状态 (100 * 5 = 500)
        for j in range(self.num_tasks):
            task_state = np.array([
                self.task_positions[j, 0] / self.map_size,
                self.task_positions[j, 1] / self.map_size,
                self.task_urgency[j],
                self.task_remaining_time[j] / 400.0,
                self.task_processing_progress[j]
            ])
            obs_components.append(task_state)
        
        # 3. 全局统计信息 (10)
        # 安全计算待处理任务的平均紧急度
        pending_mask = self.task_status == 0
        if np.any(pending_mask):
            avg_pending_urgency = np.mean(self.task_urgency[pending_mask])
        else:
            avg_pending_urgency = 0.0  # ← 如果没有待处理任务，返回 0
    
        stats = np.array([
            self.current_step / self.max_steps,
            self.completed_tasks / self.num_tasks,
            self.failed_tasks / self.num_tasks,
            np.mean(self.uav_energy),
            np.sum(self.task_status == 0) / self.num_tasks,
            np.sum(self.task_status == 1) / self.num_tasks,
            np.sum(self.task_status == 2) / self.num_tasks,
            avg_pending_urgency,  # ← 使用安全值
            self.total_distance_traveled / (self.num_uavs * self.map_size * 2),
            0.0
        ])
        
        obs_components.append(stats)
        obs = np.concatenate(obs_components)
        
        # 确保维度正确
        if len(obs) < self.global_obs_dim:
            obs = np.pad(obs, (0, self.global_obs_dim - len(obs)))
        else:
            obs = obs[:self.global_obs_dim]
        
        return torch.from_numpy(obs).float()

    def _get_low_obs(self) -> List[torch.Tensor]:
        """构造局部观测 (每个UAV一个)"""
        low_obs_list = []
        
        for i in range(self.num_uavs):
            obs_components = []
            
            # 1. 自身状态 (6)
            self_state = np.concatenate([
                self.uav_positions[i] / self.map_size,
                self.uav_velocities[i] / self.uav_speed,
                [self.uav_energy[i]],
                [float(self.uav_task_assignments[i]) / self.num_tasks]
            ])
            obs_components.append(self_state)
            
            # 2. 最近的 5 个未完成任务 (5 * 5 = 25)
            task_obs = []
            unfinished_mask = self.task_status < 2
            
            if np.any(unfinished_mask):
                unfinished_indices = np.where(unfinished_mask)[0]
                distances = np.linalg.norm(
                    self.task_positions[unfinished_indices] - self.uav_positions[i],
                    axis=1
                )
                nearest_5 = unfinished_indices[np.argsort(distances)[:5]]
                
                if self.current_step == 1 and i == 0:
                    print(f"  [DEBUG] UAV {i} nearest tasks: {nearest_5}")
                    print(f"  [DEBUG] UAV {i} position: {self.uav_positions[i]}")
                    print(f"  [DEBUG] Task 0 position: {self.task_positions[nearest_5[0]]}")
                
                for task_idx in nearest_5:
                    relative_pos = (self.task_positions[task_idx] - self.uav_positions[i]) / self.map_size
                    dist = np.linalg.norm(relative_pos)
                    task_info = np.array([
                        relative_pos[0],
                        relative_pos[1],
                        dist,
                        self.task_urgency[task_idx],
                        self.task_remaining_time[task_idx] / 500.0
                    ])
                    task_obs.append(task_info)
            
            # 如果不足5个,填充零（注意缩进：8个空格，在 for i 循环内）
            while len(task_obs) < 5:
                task_obs.append(np.zeros(5))
            
            obs_components.extend(task_obs)
            
            # 3. 其他 UAV 信息（注意缩进：8个空格，在 for i 循环内）
            other_uavs_info = np.zeros(10)
            if self.num_uavs > 1:
                other_positions = np.delete(self.uav_positions, i, axis=0)
                relative_positions = other_positions - self.uav_positions[i]
                avg_relative_pos = np.mean(relative_positions, axis=0) / self.map_size
                min_distance = np.min(np.linalg.norm(relative_positions, axis=1))
                other_uavs_info[:2] = avg_relative_pos
                other_uavs_info[2] = min_distance / self.map_size
                other_uavs_info[3] = np.mean(np.delete(self.uav_energy, i))
            
            obs_components.append(other_uavs_info)
            
            # 4. 合并并填充（注意缩进：8个空格，在 for i 循环内）
            obs = np.concatenate(obs_components)
            
            if len(obs) < self.low_obs_dim:
                obs = np.pad(obs, (0, self.low_obs_dim - len(obs)))
            else:
                obs = obs[:self.low_obs_dim]
            
            low_obs_list.append(torch.from_numpy(obs).float())
        
        # 5. 返回所有 UAV 的观测（注意缩进：4个空格，在 for 循环外）
        return low_obs_list

    def step(self, low_actions: List[np.ndarray]) -> Tuple[torch.Tensor, List[torch.Tensor], float, bool]:
        """执行一步仿真"""
        self.current_step += 1
        reward = 0.0
        
        # === 1. 更新任务状态 (时间流逝) ===
        self.task_remaining_time -= self.dt
        
        # 超时任务标记为失败
        timeout_mask = (self.task_remaining_time <= 0) & (self.task_status < 2)
        self.task_status[timeout_mask] = -1
        newly_failed = np.sum(timeout_mask)
        self.failed_tasks += newly_failed
        reward -= newly_failed * 0.5
        
        if newly_failed > 0:
            print(f"    [Step {self.current_step}] {newly_failed} tasks timeout")
        
        # === 2. UAV 运动和任务处理 ===
        for i in range(self.num_uavs):
            action = low_actions[i]
            
            # 解析动作: [vx_normalized, vy_normalized, task_select, hover]
            vx = np.clip(action[0], -1, 1) * self.uav_speed
            vy = np.clip(action[1], -1, 1) * self.uav_speed
            
            # 更新速度和位置
            self.uav_velocities[i] = np.array([vx, vy])
            old_pos = self.uav_positions[i].copy()
            self.uav_positions[i] += self.uav_velocities[i] * self.dt
            
            # 边界约束
            self.uav_positions[i] = np.clip(self.uav_positions[i], 0, self.map_size)
            
            # 计算移动距离
            distance = np.linalg.norm(self.uav_positions[i] - old_pos)
            self.total_distance_traveled += distance
            
            # === 修改能量消耗公式 ===
            # 原来：energy_consumption = distance / (self.map_size * 2)
            # 问题：在 200x200 地图上，飞 50 米就消耗 50/(200*2) = 12.5% 能量，8步就耗尽
        
            # 新公式：按飞行距离占最大续航的比例消耗
            max_flight_distance = self.map_size * 5  # 假设满电可飞 5 倍地图宽度（1000米）
            energy_consumption = distance / max_flight_distance  # ← 修改这一行
            
            self.uav_energy[i] = max(0, self.uav_energy[i] - energy_consumption)
            
            # 如果能量耗尽,惩罚
            if self.uav_energy[i] <= 0:
                reward -= 0.2
                continue
            
            # === 3. 任务处理逻辑 ===
            target_idx = self.uav_task_assignments[i]
            
            # 每50步打印一次调试信息
            if self.current_step % 50 == 0:
                if target_idx != -1:
                    dist = np.linalg.norm(self.uav_positions[i] - self.task_positions[target_idx])
                    print(f"    [UAV {i}] Pos=({self.uav_positions[i][0]:.1f},{self.uav_positions[i][1]:.1f}), "
                          f"Target={target_idx}, Dist={dist:.1f}m, Action=({vx:.1f},{vy:.1f})")
                else:
                    print(f"    [UAV {i}] No assignment! Pos=({self.uav_positions[i][0]:.1f},{self.uav_positions[i][1]:.1f})")
            
            if target_idx != -1 and 0 <= target_idx < self.num_tasks:
                # 检查是否到达任务点
                dist_to_task = np.linalg.norm(self.uav_positions[i] - self.task_positions[target_idx])
                
                if dist_to_task < 15.0:  # 到达阈值 15米
                    print(f"    [ARRIVED] UAV {i} reached Task {target_idx}, dist={dist_to_task:.2f}m")
                    
                    # 开始/继续处理任务
                    if self.task_status[target_idx] == 0:
                        self.task_status[target_idx] = 1
                        print(f"    [START] Task {target_idx} processing started")
                    
                    if self.task_status[target_idx] == 1:
                        # 处理速度与紧急度成正比
                        process_speed = 0.1 * (1 + self.task_urgency[target_idx])
                        self.task_processing_progress[target_idx] += process_speed
                        
                        # 任务完成
                        if self.task_processing_progress[target_idx] >= 1.0:
                            self.task_status[target_idx] = 2
                            self.completed_tasks += 1
                            
                            # 奖励
                            base_reward = 1.0
                            urgency_bonus = self.task_urgency[target_idx] * 0.5
                            time_bonus = (self.task_remaining_time[target_idx] / 500.0) * 0.3
                            total_task_reward = base_reward + urgency_bonus + time_bonus
                            reward += total_task_reward
                            
                            print(f"    [COMPLETE] Task {target_idx} completed! Reward={total_task_reward:.2f}")
                            
                            # 清除分配
                            self.uav_task_assignments[i] = -1
                        else:
                            # 持续处理中的小奖励
                            reward += 0.01
                else:
                    # 距离惩罚 (鼓励靠近目标)
                    reward -= 0.0001 * (dist_to_task / self.map_size)
        
        # === 4. 效率奖励 ===
        if all(x != -1 for x in self.uav_task_assignments):
            reward += 0.05
        
        # === 5. 判断结束条件 ===
        done = False
        
        if self.current_step >= self.max_steps:
            done = True
            print(f"  [Episode End] Max steps reached: {self.current_step}")
        
        elif self.completed_tasks + self.failed_tasks >= self.num_tasks:
            done = True
            completion_rate = self.completed_tasks / self.num_tasks
            reward += completion_rate * 5.0
            print(f"  [Episode End] All tasks done: Completed={self.completed_tasks}, Failed={self.failed_tasks}")
        
        elif np.all(self.uav_energy <= 0.1):
            done = True
            reward -= 2.0
            print(f"  [Episode End] All UAVs out of energy")
        
        # === 6. 构造新观测 ===
        next_global_obs = self._get_global_obs()
        next_low_obs = self._get_low_obs()
        
        self.total_reward += reward
        reward = np.clip(reward, -10.0, 10.0)
        
        return next_global_obs, next_low_obs, reward, done

    def apply_high_level_allocation(self, allocation: List[int]):
        """高层策略分配任务给 UAV"""
        if isinstance(allocation, int):
            allocation = [allocation]
        
        # 如果高层只输出1个任务ID，为所有UAV分配不同任务
        if len(allocation) == 1:
            base_task = allocation[0]
            allocation = [(base_task + i) % self.num_tasks for i in range(self.num_uavs)]
        
        print(f"  [Allocation] High-level decision: {allocation}")
        
        # 获取所有未完成任务的索引
        unfinished_tasks = np.where(self.task_status < 2)[0]
        
        for i in range(min(len(allocation), self.num_uavs)):
            task_idx = int(allocation[i]) % self.num_tasks
            
            # 如果目标任务未完成，直接分配
            if self.task_status[task_idx] < 2:
                old_assignment = self.uav_task_assignments[i]
                self.uav_task_assignments[i] = task_idx
                print(f"    [Assign] UAV {i}: Task {old_assignment} -> {task_idx}")
            
            # 如果目标任务已完成，分配下一个未完成任务
            elif len(unfinished_tasks) > 0:
                # 选择离当前UAV最近的未完成任务
                distances = np.linalg.norm(
                    self.task_positions[unfinished_tasks] - self.uav_positions[i], 
                    axis=1
                )
                nearest_task = unfinished_tasks[np.argmin(distances)]
                
                old_assignment = self.uav_task_assignments[i]
                self.uav_task_assignments[i] = nearest_task
                print(f"    [Reassign] UAV {i}: Task {old_assignment} -> {nearest_task} (original {task_idx} completed)")
            
            # 如果没有未完成任务，清除分配
            else:
                self.uav_task_assignments[i] = -1
                print(f"    [Idle] UAV {i}: No unfinished tasks available")
    
    def render(self):
        """打印当前状态 (用于调试)"""
        print(f"\n=== Step {self.current_step} ===")
        print(f"Completed: {self.completed_tasks}/{self.num_tasks}")
        print(f"Failed: {self.failed_tasks}/{self.num_tasks}")
        print(f"Average Energy: {np.mean(self.uav_energy):.2f}")
        print(f"Total Reward: {self.total_reward:.2f}")
