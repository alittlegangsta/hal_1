"""
特征工程模块 - 生成机器学习训练所需的特征和标签
"""
import numpy as np
from typing import Dict, Tuple, List
import logging
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.signal_processor = SignalProcessor()
        
        # 方位接收器到角度的映射 (45度扇区)
        self.azimuth_mapping = {
            'A': (-22.5, 22.5),     # 0度
            'B': (22.5, 67.5),     # 45度
            'C': (67.5, 112.5),    # 90度
            'D': (112.5, 157.5),   # 135度
            'E': (157.5, 202.5),   # 180度
            'F': (202.5, 247.5),   # 225度
            'G': (247.5, 292.5),   # 270度
            'H': (292.5, 337.5)    # 315度
        }
        
        # 将负角度转换为正角度
        for side in self.azimuth_mapping:
            min_angle, max_angle = self.azimuth_mapping[side]
            if min_angle < 0:
                min_angle += 360
            if max_angle < 0:
                max_angle += 360
            self.azimuth_mapping[side] = (min_angle, max_angle)
    
    def generate_training_data(self, cast_data: Dict, xsilmr_data: Dict) -> Tuple[List, List, List]:
        """
        生成训练数据
        
        Args:
            cast_data: CAST超声数据
            xsilmr_data: XSILMR声波数据
            
        Returns:
            特征列表(图像), 特征列表(数值), 标签列表
        """
        logger.info("开始生成训练数据...")
        
        X_images = []  # 尺度图特征
        X_vectors = []  # 数值特征
        y_labels = []   # 窜槽比例标签
        
        # 遍历所有接收器和方位
        for receiver_idx in sorted(xsilmr_data.keys()):
            receiver_data = xsilmr_data[receiver_idx]
            absolute_depths = receiver_data['AbsoluteDepth']
            
            logger.info(f"处理接收器 {receiver_idx}...")
            
            for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                side_key = f'Side{side}'
                if side_key not in receiver_data:
                    continue
                
                waveforms = receiver_data[side_key]  # (1024, n_depth)
                n_time, n_depth = waveforms.shape
                
                # 应用高通滤波
                filtered_waveforms = self.signal_processor.apply_highpass_filter(waveforms)
                
                # 为每个深度点生成特征和标签
                actual_depth_count = min(n_depth, len(absolute_depths))
                for depth_idx in range(actual_depth_count):
                    depth_value = absolute_depths[depth_idx]
                    waveform = filtered_waveforms[:, depth_idx]
                    
                    # 生成图像特征 (尺度图)
                    scalogram = self.signal_processor.generate_scalogram(waveform)
                    X_images.append(scalogram)
                    
                    # 生成数值特征
                    physical_features = self.signal_processor.extract_physical_features(waveform)
                    feature_vector = self._dict_to_vector(physical_features)
                    X_vectors.append(feature_vector)
                    
                    # 生成标签 (窜槽比例)
                    channeling_ratio = self._calculate_channeling_ratio(
                        cast_data, depth_value, side)
                    y_labels.append(channeling_ratio)
                    
                if len(X_images) % 1000 == 0:
                    logger.info(f"已处理 {len(X_images)} 个样本...")
        
        logger.info(f"训练数据生成完成: 共 {len(X_images)} 个样本")
        return X_images, X_vectors, y_labels
    
    def generate_enhanced_training_data(self, cast_data: Dict, xsilmr_data: Dict) -> Tuple[List, List, List]:
        """
        生成增强版训练数据 (使用阵列信号处理)
        
        Args:
            cast_data: CAST超声数据
            xsilmr_data: XSILMR声波数据
            
        Returns:
            特征列表(图像), 特征列表(数值), 标签列表
        """
        logger.info("开始生成增强版训练数据...")
        
        X_images = []
        X_vectors = []
        y_labels = []
        
        # 为每个方位角分别处理
        for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            logger.info(f"处理方位 {side}...")
            
            # 收集该方位角下所有接收器的波形
            side_waveforms = {}
            common_depths = None
            
            for receiver_idx in sorted(xsilmr_data.keys()):
                receiver_data = xsilmr_data[receiver_idx]
                side_key = f'Side{side}'
                
                if side_key in receiver_data:
                    waveforms = receiver_data[side_key]
                    filtered_waveforms = self.signal_processor.apply_highpass_filter(waveforms)
                    side_waveforms[receiver_idx] = filtered_waveforms
                    
                    if common_depths is None:
                        common_depths = receiver_data['AbsoluteDepth']
            
            if len(side_waveforms) < 2:
                logger.warning(f"方位 {side} 的接收器数量不足，跳过")
                continue
            
            # 执行慢度-时间相干性分析
            local_slowness, coherent_waveforms, quality_metrics = self.signal_processor.slowness_time_coherence(
                side_waveforms)
            
            # 计算衰减率
            attenuation = self.signal_processor.calculate_attenuation(coherent_waveforms)
            
            # 为每个接收器的每个深度点生成特征
            for receiver_idx in sorted(coherent_waveforms.keys()):
                coherent_wave = coherent_waveforms[receiver_idx]
                n_time, n_depth = coherent_wave.shape
                
                actual_depth_count = min(n_depth, len(common_depths))
                for depth_idx in range(actual_depth_count):
                    depth_value = common_depths[depth_idx]
                    waveform = coherent_wave[:, depth_idx]
                    
                    # 图像特征 (基于相干滤波后的波形)
                    scalogram = self.signal_processor.generate_scalogram(waveform)
                    X_images.append(scalogram)
                    
                    # 数值特征 (物理特征 + 慢度 + 衰减率)
                    physical_features = self.signal_processor.extract_physical_features(waveform)
                    
                    # 添加慢度特征
                    receiver_list = sorted(side_waveforms.keys())
                    if receiver_idx in receiver_list:
                        r_idx = receiver_list.index(receiver_idx)
                        if r_idx < local_slowness.shape[0]:
                            physical_features['local_slowness'] = local_slowness[r_idx, depth_idx]
                        else:
                            physical_features['local_slowness'] = 0.0
                    else:
                        physical_features['local_slowness'] = 0.0
                    
                    # 添加衰减率特征
                    if r_idx < attenuation.shape[0]:
                        physical_features['attenuation_rate'] = attenuation[r_idx, depth_idx]
                    else:
                        physical_features['attenuation_rate'] = 0.0
                    
                    feature_vector = self._dict_to_vector(physical_features)
                    X_vectors.append(feature_vector)
                    
                    # 生成标签
                    channeling_ratio = self._calculate_channeling_ratio(
                        cast_data, depth_value, side)
                    y_labels.append(channeling_ratio)
                    
                if len(X_images) % 1000 == 0:
                    logger.info(f"已处理 {len(X_images)} 个样本...")
        
        logger.info(f"增强版训练数据生成完成: 共 {len(X_images)} 个样本")
        return X_images, X_vectors, y_labels
    
    def _calculate_channeling_ratio(self, cast_data: Dict, depth: float, side: str) -> float:
        """
        计算指定深度和方位的窜槽比例
        
        Args:
            cast_data: CAST数据
            depth: 深度值
            side: 方位接收器标识
            
        Returns:
            窜槽比例 (0-1)
        """
        # 获取方位角范围
        if side not in self.azimuth_mapping:
            return 0.0
        
        min_angle, max_angle = self.azimuth_mapping[side]
        
        # 定义深度窗口 (±0.25 ft)
        depth_window = 0.25
        depth_mask = (cast_data['Depth'] >= depth - depth_window) & \
                    (cast_data['Depth'] <= depth + depth_window)
        
        if not np.any(depth_mask):
            return 0.0
        
        # CAST数据的角度索引 (每2度一个，共180个)
        angles = np.arange(0, 360, 2)  # 0, 2, 4, ..., 358
        
        # 处理跨越0度的情况
        if min_angle > max_angle:  # 跨越0度
            angle_mask = (angles >= min_angle) | (angles <= max_angle)
        else:
            angle_mask = (angles >= min_angle) & (angles <= max_angle)
        
        # 提取对应区域的Zc值
        zc_region = cast_data['Zc'][angle_mask, :][:, depth_mask]
        
        if zc_region.size == 0:
            return 0.0
        
        # 计算窜槽比例 (Zc < 2.5 的比例)
        channeling_points = np.sum(zc_region < 2.5)
        total_points = zc_region.size
        
        return channeling_points / total_points if total_points > 0 else 0.0
    
    def _dict_to_vector(self, feature_dict: Dict) -> np.ndarray:
        """
        将特征字典转换为向量
        
        Args:
            feature_dict: 特征字典
            
        Returns:
            特征向量
        """
        # 定义特征顺序
        feature_order = [
            'max_amplitude', 'rms_amplitude', 'energy', 'zero_crossings',
            'dominant_frequency', 'spectral_centroid', 'local_slowness', 'attenuation_rate'
        ]
        
        vector = []
        for feature_name in feature_order:
            if feature_name in feature_dict:
                value = feature_dict[feature_name]
                # 处理可能的无穷大或NaN值
                if np.isfinite(value):
                    vector.append(float(value))
                else:
                    vector.append(0.0)
            else:
                vector.append(0.0)
        
        return np.array(vector)
    
    def prepare_new_data(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        为新数据准备特征 (用于可逆应用)
        
        Args:
            waveform: 新的波形数据
            
        Returns:
            图像特征, 数值特征
        """
        # 应用相同的预处理
        filtered_waveform = self.signal_processor.apply_highpass_filter(waveform)
        
        # 生成特征
        scalogram = self.signal_processor.generate_scalogram(filtered_waveform)
        physical_features = self.signal_processor.extract_physical_features(filtered_waveform)
        
        # 对于新数据，慢度和衰减率设为0（需要阵列数据才能计算）
        physical_features['local_slowness'] = 0.0
        physical_features['attenuation_rate'] = 0.0
        
        feature_vector = self._dict_to_vector(physical_features)
        
        return scalogram, feature_vector 