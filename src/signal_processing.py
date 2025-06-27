"""
信号处理模块 - 包含滤波和阵列信号处理功能
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import pywt
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class SignalProcessor:
    """声波信号处理器"""
    
    def __init__(self, dt: float = 1e-5, cutoff_freq: float = 1000.0):
        """
        初始化信号处理器
        
        Args:
            dt: 采样时间间隔 (默认10微秒)
            cutoff_freq: 高通滤波截止频率 (默认1000Hz)
        """
        self.dt = dt
        self.fs = 1.0 / dt  # 采样频率
        self.cutoff_freq = cutoff_freq
        
        # 设计4阶巴特沃斯高通滤波器
        nyquist = self.fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(4, normalized_cutoff, btype='high')
        
        logger.info(f"信号处理器初始化: 采样频率={self.fs}Hz, 截止频率={cutoff_freq}Hz")
    
    def _calculate_robust_slowness(self, depth_traces: np.ndarray, receiver_spacing: float) -> Tuple[float, float]:
        """
        使用线性回归方法计算稳健的慢度值
        
        Args:
            depth_traces: 窗口内所有接收器的波形 (n_receivers, n_time)
            receiver_spacing: 接收器间距
            
        Returns:
            slowness: 计算得到的慢度值
            r_squared: 线性拟合的R²值，用于评估拟合质量
        """
        n_traces = len(depth_traces)
        
        if n_traces < 3:
            # 不足3个点，无法进行稳健的线性拟合
            if n_traces == 2:
                # 使用简单的两点法
                trace1, trace2 = depth_traces[0], depth_traces[1]
                correlation = np.correlate(trace1, trace2, mode='full')
                max_corr_idx = np.argmax(np.abs(correlation))
                time_delay = (max_corr_idx - len(trace1) + 1) * self.dt
                slowness = time_delay / receiver_spacing if receiver_spacing > 0 else 0.0
                return slowness, 0.0  # R²设为0表示质量较低
            else:
                return 0.0, 0.0
        
        # 使用窗口中心作为参考波形
        ref_idx = n_traces // 2
        reference_trace = depth_traces[ref_idx]
        
        # 计算所有波形相对于参考波形的时延
        delays = []
        distances = []
        correlations = []  # 存储相关系数用于质量控制
        
        for j, trace in enumerate(depth_traces):
            # 计算互相关得到时延
            correlation = np.correlate(reference_trace, trace, mode='full')
            max_corr_val = np.max(np.abs(correlation))
            max_corr_idx = np.argmax(np.abs(correlation))
            delay = (max_corr_idx - len(reference_trace) + 1) * self.dt
            
            # 计算相对于参考位置的距离
            distance = (j - ref_idx) * receiver_spacing
            
            # 计算归一化相关系数作为质量指标
            norm_corr = max_corr_val / (np.linalg.norm(reference_trace) * np.linalg.norm(trace) + 1e-10)
            
            delays.append(delay)
            distances.append(distance)
            correlations.append(norm_corr)
        
        # 转换为numpy数组
        distances = np.array(distances)
        delays = np.array(delays)
        correlations = np.array(correlations)
        
        # 质量控制：移除相关性太低的点
        quality_threshold = 0.3  # 相关系数阈值
        valid_mask = correlations >= quality_threshold
        
        if np.sum(valid_mask) < 3:
            # 有效点数不够，降低阈值重试
            quality_threshold = 0.1
            valid_mask = correlations >= quality_threshold
        
        if np.sum(valid_mask) < 2:
            # 仍然不够，使用所有点
            valid_mask = np.ones(len(distances), dtype=bool)
        
        # 使用有效点进行线性回归
        valid_distances = distances[valid_mask]
        valid_delays = delays[valid_mask]
        
        if len(valid_distances) < 2 or np.std(valid_distances) < 1e-6:
            return 0.0, 0.0
        
        # 线性回归: delay = slowness * distance + intercept
        n = len(valid_distances)
        sum_x = np.sum(valid_distances)
        sum_y = np.sum(valid_delays)
        sum_xy = np.sum(valid_distances * valid_delays)
        sum_x2 = np.sum(valid_distances * valid_distances)
        sum_y2 = np.sum(valid_delays * valid_delays)
        
        # 计算斜率(慢度)和截距
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) > 1e-10:
            slowness = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slowness * sum_x) / n
            
            # 计算R²值评估拟合质量
            y_pred = slowness * valid_distances + intercept
            ss_tot = np.sum((valid_delays - np.mean(valid_delays))**2)
            ss_res = np.sum((valid_delays - y_pred)**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            r_squared = max(0.0, min(1.0, r_squared))  # 限制在[0,1]范围内
        else:
            slowness = 0.0
            r_squared = 0.0
        
        return slowness, r_squared
    
    def apply_highpass_filter(self, waveform: np.ndarray) -> np.ndarray:
        """
        应用高通滤波器
        
        Args:
            waveform: 输入波形 (时间 x 深度)
            
        Returns:
            滤波后的波形
        """
        if waveform.ndim == 1:
            return filtfilt(self.b, self.a, waveform)
        else:
            # 对每一列(深度点)应用滤波
            filtered = np.zeros_like(waveform)
            for col in range(waveform.shape[1]):
                filtered[:, col] = filtfilt(self.b, self.a, waveform[:, col])
            return filtered
    
    def slowness_time_coherence(self, waveforms: Dict, window_size: int = 5) -> Tuple[np.ndarray, Dict, Optional[np.ndarray]]:
        """
        慢度-时间相干性分析 (使用稳健的线性回归方法)
        
        Args:
            waveforms: 多个接收器的波形字典 {receiver_id: waveform}
            window_size: 滑动窗口大小
            
        Returns:
            局部慢度数组, 相干滤波后的波形字典, R²质量指标数组(可选)
        """
        logger.info("执行慢度-时间相干性分析...")
        
        # 获取所有接收器ID并排序
        receiver_ids = sorted(waveforms.keys())
        n_receivers = len(receiver_ids)
        
        if n_receivers < window_size:
            logger.warning(f"接收器数量({n_receivers})小于窗口大小({window_size})")
            window_size = n_receivers
        
        # 假设所有接收器有相同的时间和深度维度
        first_key = receiver_ids[0]
        n_time, n_depth = waveforms[first_key].shape
        
        # 初始化输出
        local_slowness = np.zeros((n_receivers, n_depth))
        quality_metrics = np.zeros((n_receivers, n_depth))  # 存储R²值
        coherent_waveforms = {}
        
        # 接收器间距 (0.5 ft)
        receiver_spacing = 0.5
        
        for i, receiver_id in enumerate(receiver_ids):
            # 为每个接收器创建滑动窗口
            start_idx = max(0, i - window_size//2)
            end_idx = min(n_receivers, start_idx + window_size)
            window_receivers = receiver_ids[start_idx:end_idx]
            
            # 提取窗口内的波形
            window_waveforms = []
            for wr_id in window_receivers:
                if wr_id in waveforms:
                    window_waveforms.append(waveforms[wr_id])
            
            if len(window_waveforms) < 2:
                # 不足够的接收器进行相干分析，使用原始波形
                coherent_waveforms[receiver_id] = waveforms[receiver_id]
                local_slowness[i, :] = 0.0
                continue
            
            window_array = np.stack(window_waveforms, axis=0)  # (n_window, n_time, n_depth)
            
            # 对每个深度点进行处理
            coherent_wave = np.zeros((n_time, n_depth))
            
            for depth_idx in range(n_depth):
                depth_traces = window_array[:, :, depth_idx]  # (n_window, n_time)
                
                # 简化的相干处理：计算窗口内波形的相关性
                if len(window_waveforms) >= 2:
                    # 寻找最大相干的时延
                    reference_trace = depth_traces[len(depth_traces)//2]  # 使用中间接收器作为参考
                    coherent_trace = np.zeros_like(reference_trace, dtype=np.float64)
                    
                    for trace in depth_traces:
                        # 互相关计算时延
                        correlation = np.correlate(reference_trace, trace, mode='full')
                        max_corr_idx = np.argmax(np.abs(correlation))
                        delay = max_corr_idx - len(reference_trace) + 1
                        
                        # 应用时延校正后叠加
                        if delay != 0:
                            if delay > 0:
                                aligned_trace = np.concatenate([np.zeros(delay), trace[:-delay]])
                            else:
                                aligned_trace = np.concatenate([trace[-delay:], np.zeros(-delay)])
                        else:
                            aligned_trace = trace
                            
                        coherent_trace += aligned_trace[:len(coherent_trace)]
                    
                    coherent_trace /= len(depth_traces)
                    coherent_wave[:, depth_idx] = coherent_trace
                    
                    # 计算局部慢度 - 使用稳健的线性回归方法
                    slowness, r_squared = self._calculate_robust_slowness(depth_traces, receiver_spacing)
                    local_slowness[i, depth_idx] = slowness
                    quality_metrics[i, depth_idx] = r_squared
                else:
                    coherent_wave[:, depth_idx] = depth_traces[0]
                    local_slowness[i, depth_idx] = 0.0
            
            coherent_waveforms[receiver_id] = coherent_wave
        
        logger.info("慢度-时间相干性分析完成")
        return local_slowness, coherent_waveforms, quality_metrics
    
    def calculate_attenuation(self, waveforms: Dict) -> np.ndarray:
        """
        计算局部衰减率
        
        Args:
            waveforms: 接收器波形字典
            
        Returns:
            衰减率数组 (n_receivers, n_depth)
        """
        logger.info("计算局部衰减率...")
        
        receiver_ids = sorted(waveforms.keys())
        n_receivers = len(receiver_ids)
        
        if n_receivers < 2:
            logger.warning("接收器数量不足，无法计算衰减率")
            return np.zeros((1, waveforms[receiver_ids[0]].shape[1]))
        
        first_key = receiver_ids[0]
        n_depth = waveforms[first_key].shape[1]
        attenuation = np.zeros((n_receivers - 1, n_depth))
        
        # 接收器间距
        delta_z = 0.5  # ft
        
        for i in range(n_receivers - 1):
            receiver1_id = receiver_ids[i]
            receiver2_id = receiver_ids[i + 1]
            
            wave1 = waveforms[receiver1_id]
            wave2 = waveforms[receiver2_id]
            
            # 计算每个深度点的最大幅值
            max_amp1 = np.max(np.abs(wave1), axis=0)
            max_amp2 = np.max(np.abs(wave2), axis=0)
            
            # 避免除零和取对数的问题
            ratio = np.where((max_amp1 > 1e-10) & (max_amp2 > 1e-10), 
                           max_amp1 / max_amp2, 1.0)
            
            # 计算衰减率: α = log(A1/A2) / Δz
            with np.errstate(divide='ignore', invalid='ignore'):
                attenuation[i, :] = np.log(ratio) / delta_z
                attenuation[i, :] = np.where(np.isfinite(attenuation[i, :]), 
                                           attenuation[i, :], 0.0)
        
        logger.info("局部衰减率计算完成")
        return attenuation
    
    def generate_scalogram(self, waveform: np.ndarray, scales: np.ndarray = None) -> np.ndarray:
        """
        生成连续小波变换尺度图
        
        Args:
            waveform: 输入波形 (1D)
            scales: 小波尺度数组
            
        Returns:
            尺度图 (scales, time)
        """
        if scales is None:
            scales = np.arange(1, 128)
        
        # 使用Morlet小波
        coefficients, _ = pywt.cwt(waveform, scales, 'morl', sampling_period=self.dt)
        
        # 返回幅值
        return np.abs(coefficients)
    
    def extract_physical_features(self, waveform: np.ndarray) -> Dict:
        """
        提取波形的物理特征
        
        Args:
            waveform: 输入波形 (1D或2D)
            
        Returns:
            特征字典
        """
        features = {}
        
        if waveform.ndim == 1:
            # 1D波形特征
            features['max_amplitude'] = np.max(np.abs(waveform))
            features['rms_amplitude'] = np.sqrt(np.mean(waveform**2))
            features['energy'] = np.sum(waveform**2)
            features['zero_crossings'] = np.sum(np.diff(np.sign(waveform)) != 0)
            
            # 频域特征
            fft = np.fft.fft(waveform)
            freqs = np.fft.fftfreq(len(waveform), self.dt)
            power_spectrum = np.abs(fft)**2
            
            features['dominant_frequency'] = freqs[np.argmax(power_spectrum[:len(power_spectrum)//2])]
            features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2])
            
        else:
            # 2D波形特征 (对每个深度点计算，然后取统计量)
            max_amps = np.max(np.abs(waveform), axis=0)
            features['max_amplitude_mean'] = np.mean(max_amps)
            features['max_amplitude_std'] = np.std(max_amps)
            
            rms_amps = np.sqrt(np.mean(waveform**2, axis=0))
            features['rms_amplitude_mean'] = np.mean(rms_amps)
            features['rms_amplitude_std'] = np.std(rms_amps)
        
        return features 