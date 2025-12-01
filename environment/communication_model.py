"""
UAV通信模型
基于论文中的通信模型，包含视距/非视距概率、路径损耗、信噪比和数据传输速率计算
参考论文中的公式：Eq. (1)-(5)
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional, Union, List
import warnings
from dataclasses import dataclass
from enum import Enum


class EnvironmentType(Enum):
    """环境类型枚举"""
    URBAN = "urban"          # 城市环境
    SUBURBAN = "suburban"    # 郊区环境
    RURAL = "rural"          # 农村环境
    DENSE_URBAN = "dense_urban"  # 密集城区


@dataclass
class CommunicationConfig:
    """通信配置"""
    # 基本参数
    frequency: float = 2.4e9  # 载波频率 (Hz)
    bandwidth: float = 10e6   # 带宽 (Hz)
    tx_power: float = 20.0    # 发射功率 (dBm)
    noise_power: float = -174.0  # 噪声功率 (dBm)
    uav_height: float = 100.0  # UAV飞行高度 (m)
    
    # 环境参数（基于3GPP模型）
    environment: EnvironmentType = EnvironmentType.URBAN
    
    # 视距概率参数（环境相关）
    alpha_1: float = 9.61    # 城市环境默认值
    alpha_2: float = 0.16    # 城市环境默认值
    
    # 路径损耗参数
    path_loss_exponent_los: float = 2.0    # 视距路径损耗指数
    path_loss_exponent_nlos: float = 3.5   # 非视距路径损耗指数
    additional_loss_los: float = 2.0       # 视距额外损耗 (dB)
    additional_loss_nlos: float = 23.0     # 非视距额外损耗 (dB)
    
    # 阴影衰落参数
    shadowing_std_los: float = 4.0     # 视距阴影衰落标准差 (dB)
    shadowing_std_nlos: float = 8.0    # 非视距阴影衰落标准差 (dB)
    
    # 多径参数
    rayleigh_fading: bool = True       # 是否考虑瑞利衰落
    rician_k_factor: float = 10.0      # 莱斯K因子（仅LOS）
    
    # QoS参数
    snr_threshold: float = 10.0        # 最小SNR阈值 (dB)
    ber_target: float = 1e-6           # 目标误码率
    max_doppler_shift: float = 100.0   # 最大多普勒频移 (Hz)
    
    # OFDMA参数
    num_subcarriers: int = 1024        # 子载波数量
    subcarrier_spacing: float = 15e3   # 子载波间隔 (Hz)
    cyclic_prefix_ratio: float = 0.07  # 循环前缀比例


class CommunicationChannel:
    """通信信道模型"""
    
    def __init__(self, config: Optional[CommunicationConfig] = None):
        """
        初始化通信信道
        
        Args:
            config: 通信配置
        """
        self.config = config or CommunicationConfig()
        self._set_environment_params()
        
        # 物理常数
        self.speed_of_light = 3e8  # 光速 (m/s)
        
    def _set_environment_params(self):
        """根据环境类型设置参数"""
        env_params = {
            EnvironmentType.URBAN: {
                'alpha_1': 9.61,
                'alpha_2': 0.16,
                'additional_loss_los': 2.0,
                'additional_loss_nlos': 23.0,
                'shadowing_std_los': 4.0,
                'shadowing_std_nlos': 8.0
            },
            EnvironmentType.SUBURBAN: {
                'alpha_1': 7.14,
                'alpha_2': 0.14,
                'additional_loss_los': 1.5,
                'additional_loss_nlos': 20.0,
                'shadowing_std_los': 3.0,
                'shadowing_std_nlos': 6.0
            },
            EnvironmentType.RURAL: {
                'alpha_1': 4.88,
                'alpha_2': 0.12,
                'additional_loss_los': 1.0,
                'additional_loss_nlos': 15.0,
                'shadowing_std_los': 2.0,
                'shadowing_std_nlos': 4.0
            },
            EnvironmentType.DENSE_URBAN: {
                'alpha_1': 12.5,
                'alpha_2': 0.18,
                'additional_loss_los': 3.0,
                'additional_loss_nlos': 28.0,
                'shadowing_std_los': 5.0,
                'shadowing_std_nlos': 10.0
            }
        }
        
        if self.config.environment in env_params:
            params = env_params[self.config.environment]
            for key, value in params.items():
                setattr(self.config, key, value)
                
    def calculate_elevation_angle(self, horizontal_distance: float, 
                                 uav_height: Optional[float] = None) -> float:
        """
        计算仰角
        
        Args:
            horizontal_distance: 水平距离 (m)
            uav_height: UAV高度 (m)，如果为None则使用配置中的高度
            
        Returns:
            仰角 (弧度)
        """
        height = uav_height if uav_height is not None else self.config.uav_height
        
        if horizontal_distance <= 0:
            return np.pi / 2  # 垂直向下
            
        return np.arctan(height / horizontal_distance)
        
    def calculate_los_probability(self, horizontal_distance: float, 
                                 uav_height: Optional[float] = None,
                                 elevation_angle: Optional[float] = None) -> float:
        """
        计算视距概率 (Eq. 1 in paper)
        
        Args:
            horizontal_distance: 水平距离 (m)
            uav_height: UAV高度 (m)
            elevation_angle: 仰角 (弧度)，如果提供则直接使用
            
        Returns:
            视距概率 (0-1)
        """
        if elevation_angle is None:
            if uav_height is None:
                uav_height = self.config.uav_height
            elevation_angle = self.calculate_elevation_angle(horizontal_distance, uav_height)
            
        # 将弧度转换为角度
        elevation_deg = np.degrees(elevation_angle)
        
        # 计算视距概率 (论文中的Eq. 1)
        # Pr_LoS = 1 / (1 + alpha1 * exp(-alpha2 * (theta - alpha1)))
        denominator = 1 + self.config.alpha_1 * np.exp(-self.config.alpha_2 * 
                                                      (elevation_deg - self.config.alpha_1))
        probability = 1.0 / denominator
        
        # 确保概率在[0, 1]范围内
        return np.clip(probability, 0.0, 1.0)
        
    def calculate_path_loss_los(self, distance: float, 
                               frequency: Optional[float] = None) -> float:
        """
        计算视距路径损耗 (Eq. 2 in paper)
        
        Args:
            distance: 通信距离 (m)
            frequency: 载波频率 (Hz)，如果为None则使用配置中的频率
            
        Returns:
            路径损耗 (dB)
        """
        freq = frequency if frequency is not None else self.config.frequency
        
        # 自由空间路径损耗
        fspl = 20 * np.log10(distance) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / self.speed_of_light)
        
        # 加上额外损耗
        path_loss = fspl + self.config.additional_loss_los
        
        return path_loss
        
    def calculate_path_loss_nlos(self, distance: float, 
                                frequency: Optional[float] = None) -> float:
        """
        计算非视距路径损耗 (Eq. 3 in paper)
        
        Args:
            distance: 通信距离 (m)
            frequency: 载波频率 (Hz)
            
        Returns:
            路径损耗 (dB)
        """
        freq = frequency if frequency is not None else self.config.frequency
        
        # NLOS路径损耗
        path_loss = (self.config.path_loss_exponent_nlos * 10 * np.log10(distance) +
                    20 * np.log10(freq) + 20 * np.log10(4 * np.pi / self.speed_of_light) +
                    self.config.additional_loss_nlos)
        
        return path_loss
        
    def calculate_total_path_loss(self, horizontal_distance: float,
                                 uav_height: Optional[float] = None,
                                 include_shadowing: bool = True) -> Dict[str, float]:
        """
        计算总路径损耗 (Eq. 4 in paper)
        
        Args:
            horizontal_distance: 水平距离 (m)
            uav_height: UAV高度 (m)
            include_shadowing: 是否包含阴影衰落
            
        Returns:
            包含详细信息的字典
        """
        height = uav_height if uav_height is not None else self.config.uav_height
        
        # 计算斜距
        slant_distance = np.sqrt(horizontal_distance**2 + height**2)
        
        # 计算仰角和视距概率
        elevation_angle = self.calculate_elevation_angle(horizontal_distance, height)
        los_probability = self.calculate_los_probability(horizontal_distance, height, elevation_angle)
        nlos_probability = 1.0 - los_probability
        
        # 计算LOS和NLOS路径损耗
        path_loss_los = self.calculate_path_loss_los(slant_distance)
        path_loss_nlos = self.calculate_path_loss_nlos(slant_distance)
        
        # 添加阴影衰落
        if include_shadowing:
            shadowing_los = np.random.normal(0, self.config.shadowing_std_los)
            shadowing_nlos = np.random.normal(0, self.config.shadowing_std_nlos)
            path_loss_los += shadowing_los
            path_loss_nlos += shadowing_nlos
            
        # 计算平均路径损耗
        avg_path_loss = (los_probability * path_loss_los + 
                        nlos_probability * path_loss_nlos)
        
        return {
            'slant_distance': slant_distance,
            'elevation_angle_deg': np.degrees(elevation_angle),
            'los_probability': los_probability,
            'path_loss_los_db': path_loss_los,
            'path_loss_nlos_db': path_loss_nlos,
            'avg_path_loss_db': avg_path_loss,
            'include_shadowing': include_shadowing
        }
        
    def calculate_snr(self, horizontal_distance: float,
                     tx_power: Optional[float] = None,
                     uav_height: Optional[float] = None,
                     include_fading: bool = True) -> Dict[str, float]:
        """
        计算接收信噪比 (SNR)
        
        Args:
            horizontal_distance: 水平距离 (m)
            tx_power: 发射功率 (dBm)
            uav_height: UAV高度 (m)
            include_fading: 是否包含多径衰落
            
        Returns:
            包含SNR详细信息的字典
        """
        # 获取参数
        tx_pwr = tx_power if tx_power is not None else self.config.tx_power
        height = uav_height if uav_height is not None else self.config.uav_height
        
        # 计算路径损耗
        path_loss_info = self.calculate_total_path_loss(horizontal_distance, height, True)
        avg_path_loss = path_loss_info['avg_path_loss_db']
        
        # 计算接收功率
        rx_power = tx_pwr - avg_path_loss
        
        # 添加多径衰落（瑞利或莱斯）
        if include_fading and self.config.rayleigh_fading:
            if path_loss_info['los_probability'] > 0.7:  # 高LOS概率，使用莱斯衰落
                # 莱斯衰落：LOS分量 + NLOS分量
                k = self.config.rician_k_factor * path_loss_info['los_probability']
                fading_db = self._calculate_rician_fading(k)
            else:  # 低LOS概率，使用瑞利衰落
                fading_db = self._calculate_rayleigh_fading()
            rx_power += fading_db
            
        # 计算SNR
        snr = rx_power - self.config.noise_power
        
        # 计算SNR线性值
        snr_linear = 10**(snr / 10.0)
        
        # 计算中断概率（SNR低于阈值）
        outage = 1.0 if snr < self.config.snr_threshold else 0.0
        
        return {
            'tx_power_db': tx_pwr,
            'rx_power_db': rx_power,
            'path_loss_db': avg_path_loss,
            'noise_power_db': self.config.noise_power,
            'snr_db': snr,
            'snr_linear': snr_linear,
            'outage': outage,
            'meets_qos': snr >= self.config.snr_threshold,
            **path_loss_info
        }
        
    def calculate_data_rate(self, horizontal_distance: float,
                           bandwidth: Optional[float] = None,
                           tx_power: Optional[float] = None,
                           uav_height: Optional[float] = None,
                           modulation: str = 'adaptive') -> Dict[str, float]:
        """
        计算数据传输速率 (Eq. 5 in paper)
        
        Args:
            horizontal_distance: 水平距离 (m)
            bandwidth: 带宽 (Hz)
            tx_power: 发射功率 (dBm)
            uav_height: UAV高度 (m)
            modulation: 调制方式 ('adaptive', 'qpsk', '16qam', '64qam')
            
        Returns:
            包含数据速率详细信息的字典
        """
        # 获取SNR信息
        snr_info = self.calculate_snr(horizontal_distance, tx_power, uav_height, True)
        
        # 获取带宽
        bw = bandwidth if bandwidth is not None else self.config.bandwidth
        
        if not snr_info['meets_qos']:
            # 不满足QoS，数据速率为0
            spectral_efficiency = 0.0
            data_rate = 0.0
            effective_modulation = 'none'
        else:
            # 根据SNR选择调制方式
            spectral_efficiency, effective_modulation = self._select_modulation(
                snr_info['snr_db'], modulation
            )
            
            # 计算数据速率 (香农公式近似)
            if modulation == 'adaptive':
                # 使用实际调制方案的频谱效率
                data_rate = bw * spectral_efficiency
            else:
                # 使用香农容量
                capacity = bw * np.log2(1 + snr_info['snr_linear'])
                data_rate = capacity * 0.8  # 实际效率因子
        
        # 考虑OFDMA效率
        ofdma_efficiency = self._calculate_ofdma_efficiency()
        effective_data_rate = data_rate * ofdma_efficiency
        
        return {
            'snr_db': snr_info['snr_db'],
            'bandwidth_hz': bw,
            'spectral_efficiency': spectral_efficiency,
            'modulation': effective_modulation,
            'shannon_capacity_bps': bw * np.log2(1 + snr_info['snr_linear']),
            'theoretical_data_rate_bps': data_rate,
            'ofdma_efficiency': ofdma_efficiency,
            'effective_data_rate_bps': effective_data_rate,
            'effective_data_rate_mbps': effective_data_rate / 1e6,
            'meets_qos': snr_info['meets_qos'],
            'outage': snr_info['outage']
        }
        
    def _calculate_rayleigh_fading(self) -> float:
        """计算瑞利衰落（dB）"""
        # 瑞利分布：幅度服从瑞利分布，功率服从指数分布
        # 转换为dB
        fading_power = np.random.exponential(1.0)  # 单位平均功率
        fading_db = 10 * np.log10(fading_power)
        return fading_db
        
    def _calculate_rician_fading(self, k_factor: float) -> float:
        """计算莱斯衰落（dB）"""
        # 莱斯分布：有LOS分量
        # K因子：LOS功率与NLOS功率之比
        if k_factor <= 0:
            return self._calculate_rayleigh_fading()
            
        # 计算LOS和NLOS分量
        los_amplitude = np.sqrt(k_factor / (k_factor + 1))
        nlos_amplitude = 1.0 / np.sqrt(2 * (k_factor + 1))
        
        # 生成复高斯随机变量
        nlos_i = np.random.normal(0, nlos_amplitude)
        nlos_q = np.random.normal(0, nlos_amplitude)
        
        # 总幅度
        total_amplitude = np.sqrt((los_amplitude + nlos_i)**2 + nlos_q**2)
        
        # 转换为dB
        fading_db = 20 * np.log10(total_amplitude)
        return fading_db
        
    def _select_modulation(self, snr_db: float, modulation: str) -> Tuple[float, str]:
        """
        根据SNR选择调制方式和频谱效率
        
        Args:
            snr_db: 信噪比 (dB)
            modulation: 调制方式
            
        Returns:
            (频谱效率, 调制方式)
        """
        if modulation != 'adaptive':
            # 固定调制方式
            modulations = {
                'bpsk': (0.5, 'BPSK'),
                'qpsk': (1.0, 'QPSK'),
                '16qam': (2.0, '16QAM'),
                '64qam': (3.0, '64QAM'),
                '256qam': (4.0, '256QAM')
            }
            if modulation in modulations:
                return modulations[modulation]
            else:
                warnings.warn(f"未知调制方式: {modulation}，使用QPSK")
                return (1.0, 'QPSK')
                
        # 自适应调制
        if snr_db < 6:
            return (0.5, 'BPSK')      # 需要SNR > 5 dB
        elif snr_db < 11:
            return (1.0, 'QPSK')      # 需要SNR > 10 dB
        elif snr_db < 16:
            return (2.0, '16QAM')     # 需要SNR > 15 dB
        elif snr_db < 22:
            return (3.0, '64QAM')     # 需要SNR > 21 dB
        else:
            return (4.0, '256QAM')    # 需要SNR > 27 dB
            
    def _calculate_ofdma_efficiency(self) -> float:
        """计算OFDMA效率"""
        # 考虑循环前缀开销
        cp_overhead = self.config.cyclic_prefix_ratio
        efficiency = 1.0 - cp_overhead
        
        # 考虑保护间隔等
        efficiency *= 0.95
        
        return efficiency
        
    def calculate_coverage_area(self, min_data_rate: float = 1e6,
                               uav_height: Optional[float] = None,
                               resolution: int = 50) -> Dict[str, np.ndarray]:
        """
        计算UAV的覆盖区域
        
        Args:
            min_data_rate: 最小数据速率 (bps)
            uav_height: UAV高度 (m)
            resolution: 分辨率
            
        Returns:
            包含覆盖区域信息的字典
        """
        height = uav_height if uav_height is not None else self.config.uav_height
        
        # 创建网格
        max_range = height * 3  # 假设最大覆盖范围为高度的3倍
        x = np.linspace(-max_range, max_range, resolution)
        y = np.linspace(-max_range, max_range, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 初始化结果矩阵
        data_rate_map = np.zeros((resolution, resolution))
        snr_map = np.zeros((resolution, resolution))
        los_map = np.zeros((resolution, resolution))
        coverage_map = np.zeros((resolution, resolution))
        
        # 计算每个点的性能
        for i in range(resolution):
            for j in range(resolution):
                horizontal_distance = np.sqrt(X[i, j]**2 + Y[i, j]**2)
                
                # 计算数据速率
                if horizontal_distance <= max_range:
                    rate_info = self.calculate_data_rate(horizontal_distance, 
                                                        uav_height=height)
                    snr_info = self.calculate_snr(horizontal_distance, 
                                                 uav_height=height)
                    
                    data_rate_map[i, j] = rate_info['effective_data_rate_bps']
                    snr_map[i, j] = snr_info['snr_db']
                    los_map[i, j] = snr_info['los_probability']
                    
                    # 检查是否满足覆盖要求
                    if (rate_info['effective_data_rate_bps'] >= min_data_rate and 
                        snr_info['snr_db'] >= self.config.snr_threshold):
                        coverage_map[i, j] = 1.0
                        
        return {
            'x_grid': X,
            'y_grid': Y,
            'data_rate_map': data_rate_map,
            'snr_map': snr_map,
            'los_probability_map': los_map,
            'coverage_map': coverage_map,
            'coverage_ratio': np.mean(coverage_map),
            'max_range': max_range,
            'min_data_rate': min_data_rate
        }
        
    def calculate_link_budget(self, distance: float, 
                             required_data_rate: float = 1e6) -> Dict[str, float]:
        """
        计算链路预算
        
        Args:
            distance: 通信距离 (m)
            required_data_rate: 所需数据速率 (bps)
            
        Returns:
            链路预算信息
        """
        # 计算当前配置下的数据速率
        rate_info = self.calculate_data_rate(distance)
        
        # 计算所需SNR
        required_spectral_efficiency = required_data_rate / self.config.bandwidth
        required_snr_linear = 2**required_spectral_efficiency - 1
        required_snr_db = 10 * np.log10(required_snr_linear)
        
        # 计算链路余量
        link_margin = rate_info['snr_db'] - required_snr_db
        
        # 计算最大通信距离
        max_distance = self._calculate_max_distance(required_snr_db)
        
        return {
            'distance': distance,
            'required_data_rate': required_data_rate,
            'current_data_rate': rate_info['effective_data_rate_bps'],
            'required_snr_db': required_snr_db,
            'current_snr_db': rate_info['snr_db'],
            'link_margin_db': link_margin,
            'max_distance': max_distance,
            'feasible': link_margin >= 0
        }
        
    def _calculate_max_distance(self, required_snr_db: float, 
                               max_iterations: int = 20,
                               tolerance: float = 0.1) -> float:
        """
        计算满足SNR要求的最大通信距离
        
        Args:
            required_snr_db: 所需SNR (dB)
            max_iterations: 最大迭代次数
            tolerance: 容忍误差 (dB)
            
        Returns:
            最大距离 (m)
        """
        # 使用二分法搜索最大距离
        low = 1.0  # 最小距离 1m
        high = 5000.0  # 最大距离 5km
        
        for _ in range(max_iterations):
            mid = (low + high) / 2
            snr_info = self.calculate_snr(mid)
            
            if snr_info['snr_db'] >= required_snr_db:
                low = mid  # 可以增加距离
            else:
                high = mid  # 需要减少距离
                
            if high - low < tolerance * 10:  # 距离精度
                break
                
        return low
        
    def get_channel_state_info(self, distance: float, 
                              include_fast_fading: bool = True) -> Dict[str, float]:
        """
        获取信道状态信息
        
        Args:
            distance: 通信距离 (m)
            include_fast_fading: 是否包含快衰落
            
        Returns:
            信道状态信息
        """
        # 计算大尺度衰落
        large_scale = self.calculate_total_path_loss(distance, include_shadowing=True)
        
        # 计算快衰落（如果启用）
        fast_fading_db = 0.0
        fast_fading_type = 'none'
        
        if include_fast_fading and self.config.rayleigh_fading:
            los_prob = large_scale['los_probability']
            if los_prob > 0.7:
                k = self.config.rician_k_factor * los_prob
                fast_fading_db = self._calculate_rician_fading(k)
                fast_fading_type = 'rician'
            else:
                fast_fading_db = self._calculate_rayleigh_fading()
                fast_fading_type = 'rayleigh'
                
        # 计算多普勒频移
        doppler_shift = self._calculate_doppler_shift(distance)
        
        # 计算相干时间
        coherence_time = 0.423 / doppler_shift if doppler_shift > 0 else float('inf')
        
        return {
            'large_scale_path_loss_db': large_scale['avg_path_loss_db'],
            'los_probability': large_scale['los_probability'],
            'fast_fading_db': fast_fading_db,
            'fast_fading_type': fast_fading_type,
            'doppler_shift_hz': doppler_shift,
            'coherence_time_s': coherence_time,
            'channel_variability': 'slow' if coherence_time > 0.1 else 'fast'
        }


class OFDMAScheduler:
    """OFDMA调度器"""
    
    def __init__(self, num_subcarriers: int = 1024, 
                 subcarrier_spacing: float = 15e3):
        """
        初始化OFDMA调度器
        
        Args:
            num_subcarriers: 子载波数量
            subcarrier_spacing: 子载波间隔 (Hz)
        """
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.total_bandwidth = num_subcarriers * subcarrier_spacing
        
        # 资源块分配
        self.rb_size = 12  # 每个资源块的子载波数
        self.num_rbs = num_subcarriers // self.rb_size
        
        # 用户分配
        self.user_allocations = {}
        
    def allocate_resources(self, user_demands: Dict[str, float],
                          channel_conditions: Dict[str, float]) -> Dict[str, Dict]:
        """
        分配OFDMA资源
        
        Args:
            user_demands: 用户需求 {user_id: required_rate}
            channel_conditions: 信道条件 {user_id: snr_db}
            
        Returns:
            资源分配结果
        """
        # 根据信道条件排序用户
        sorted_users = sorted(channel_conditions.items(), 
                             key=lambda x: x[1], reverse=True)
        
        allocations = {}
        remaining_rbs = self.num_rbs
        
        for user_id, snr_db in sorted_users:
            if remaining_rbs <= 0:
                break
                
            # 计算该用户的频谱效率
            if user_id not in user_demands:
                continue
                
            required_rate = user_demands[user_id]
            
            # 根据SNR计算可达到的速率
            snr_linear = 10**(snr_db / 10.0)
            spectral_efficiency = np.log2(1 + snr_linear)
            
            # 计算所需资源块
            rb_bandwidth = self.rb_size * self.subcarrier_spacing
            required_rbs = int(np.ceil(required_rate / (rb_bandwidth * spectral_efficiency)))
            allocated_rbs = min(required_rbs, remaining_rbs)
            
            # 分配资源
            if allocated_rbs > 0:
                # 选择连续的资源块（简化）
                start_rb = self.num_rbs - remaining_rbs
                end_rb = start_rb + allocated_rbs
                
                allocations[user_id] = {
                    'allocated_rbs': allocated_rbs,
                    'start_rb': start_rb,
                    'end_rb': end_rb,
                    'allocated_bandwidth': allocated_rbs * rb_bandwidth,
                    'achievable_rate': allocated_rbs * rb_bandwidth * spectral_efficiency,
                    'snr_db': snr_db,
                    'spectral_efficiency': spectral_efficiency
                }
                
                remaining_rbs -= allocated_rbs
                
        # 记录分配
        self.user_allocations = allocations
        
        return allocations
        
    def calculate_system_capacity(self, channel_conditions: Dict[str, float]) -> float:
        """
        计算系统容量
        
        Args:
            channel_conditions: 信道条件 {user_id: snr_db}
            
        Returns:
            系统总容量 (bps)
        """
        total_capacity = 0.0
        
        for user_id, snr_db in channel_conditions.items():
            snr_linear = 10**(snr_db / 10.0)
            spectral_efficiency = np.log2(1 + snr_linear)
            user_capacity = self.total_bandwidth * spectral_efficiency / len(channel_conditions)
            total_capacity += user_capacity
            
        return total_capacity


# 测试函数
def test_communication_model():
    """测试通信模型"""
    print("=" * 60)
    print("通信模型测试")
    print("=" * 60)
    
    try:
        # 创建配置
        config = CommunicationConfig(
            frequency=2.4e9,
            bandwidth=20e6,
            tx_power=23.0,
            environment=EnvironmentType.URBAN
        )
        
        # 创建信道模型
        channel = CommunicationChannel(config)
        
        print("1. 测试基本功能:")
        print("-" * 40)
        
        # 测试仰角计算
        distance = 500  # 500米水平距离
        elevation = channel.calculate_elevation_angle(distance)
        print(f"  水平距离 {distance}m, UAV高度 {config.uav_height}m")
        print(f"  仰角: {np.degrees(elevation):.1f}度")
        
        # 测试视距概率
        los_prob = channel.calculate_los_probability(distance)
        print(f"  视距概率: {los_prob:.3f}")
        
        # 测试路径损耗
        path_loss_info = channel.calculate_total_path_loss(distance)
        print(f"  平均路径损耗: {path_loss_info['avg_path_loss_db']:.1f} dB")
        
        # 测试SNR
        print(f"\n2. 测试SNR计算:")
        print("-" * 40)
        
        snr_info = channel.calculate_snr(distance)
        print(f"  发射功率: {snr_info['tx_power_db']} dBm")
        print(f"  接收功率: {snr_info['rx_power_db']:.1f} dBm")
        print(f"  噪声功率: {snr_info['noise_power_db']} dBm")
        print(f"  SNR: {snr_info['snr_db']:.1f} dB")
        print(f"  满足QoS要求: {snr_info['meets_qos']}")
        
        # 测试数据速率
        print(f"\n3. 测试数据速率计算:")
        print("-" * 40)
        
        rate_info = channel.calculate_data_rate(distance)
        print(f"  带宽: {rate_info['bandwidth_hz']/1e6:.1f} MHz")
        print(f"  频谱效率: {rate_info['spectral_efficiency']:.2f} bps/Hz")
        print(f"  调制方式: {rate_info['modulation']}")
        print(f"  香农容量: {rate_info['shannon_capacity_bps']/1e6:.2f} Mbps")
        print(f"  有效数据速率: {rate_info['effective_data_rate_mbps']:.2f} Mbps")
        
        # 测试不同距离的性能
        print(f"\n4. 测试不同距离的性能:")
        print("-" * 40)
        
        distances = [100, 300, 500, 1000]
        print("  距离(m)  SNR(dB)  速率(Mbps)  覆盖")
        print("  " + "-" * 40)
        
        for d in distances:
            rate_info = channel.calculate_data_rate(d)
            snr_info = channel.calculate_snr(d)
            coverage = "是" if rate_info['meets_qos'] else "否"
            print(f"  {d:6}   {snr_info['snr_db']:7.1f}   {rate_info['effective_data_rate_mbps']:9.2f}    {coverage}")
        
        # 测试链路预算
        print(f"\n5. 测试链路预算:")
        print("-" * 40)
        
        budget = channel.calculate_link_budget(800, 5e6)  # 800m距离，需要5Mbps
        print(f"  通信距离: {budget['distance']} m")
        print(f"  所需数据速率: {budget['required_data_rate']/1e6:.1f} Mbps")
        print(f"  当前数据速率: {budget['current_data_rate']/1e6:.1f} Mbps")
        print(f"  所需SNR: {budget['required_snr_db']:.1f} dB")
        print(f"  当前SNR: {budget['current_snr_db']:.1f} dB")
        print(f"  链路余量: {budget['link_margin_db']:.1f} dB")
        print(f"  最大通信距离: {budget['max_distance']:.0f} m")
        print(f"  链路可行: {budget['feasible']}")
        
        # 测试OFDMA调度
        print(f"\n6. 测试OFDMA调度:")
        print("-" * 40)
        
        scheduler = OFDMAScheduler(num_subcarriers=256, subcarrier_spacing=15e3)
        
        # 模拟多个用户
        user_demands = {
            'user1': 2e6,  # 2 Mbps
            'user2': 5e6,  # 5 Mbps
            'user3': 1e6   # 1 Mbps
        }
        
        channel_conditions = {
            'user1': 18.5,  # SNR in dB
            'user2': 12.3,
            'user3': 25.0
        }
        
        allocations = scheduler.allocate_resources(user_demands, channel_conditions)
        
        print("  用户ID   分配RB数   分配带宽(MHz)  可达速率(Mbps)")
        print("  " + "-" * 45)
        
        for user_id, alloc in allocations.items():
            bw_mhz = alloc['allocated_bandwidth'] / 1e6
            rate_mbps = alloc['achievable_rate'] / 1e6
            print(f"  {user_id:6}   {alloc['allocated_rbs']:8}   {bw_mhz:12.2f}   {rate_mbps:13.2f}")
        
        # 计算系统容量
        capacity = scheduler.calculate_system_capacity(channel_conditions)
        print(f"  系统总容量: {capacity/1e6:.2f} Mbps")
        
        print(f"\n{'='*60}")
        print("通信模型测试完成!")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test():
    """快速测试"""
    print("快速测试通信模型...")
    
    try:
        # 创建简单配置
        config = CommunicationConfig(
            frequency=2.4e9,
            bandwidth=10e6,
            environment=EnvironmentType.URBAN
        )
        
        # 创建信道
        channel = CommunicationChannel(config)
        
        # 测试一个距离
        distance = 300
        snr = channel.calculate_snr(distance)
        rate = channel.calculate_data_rate(distance)
        
        print(f"距离: {distance} m")
        print(f"SNR: {snr['snr_db']:.1f} dB")
        print(f"数据速率: {rate['effective_data_rate_mbps']:.2f} Mbps")
        print(f"覆盖: {'是' if rate['meets_qos'] else '否'}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        test_communication_model()