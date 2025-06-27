"""
特征工程模块 - 支持HDF5增量存储的大规模数据处理
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import gc

from .signal_processing import SignalProcessor
from .data_loader import DataLoader
from .hdf5_manager import HDF5DataManager, BatchProcessor

logger = logging.getLogger(__name__)

class IncrementalFeatureEngineer:
    """增量特征工程器 - 使用HDF5进行内存高效的特征处理"""
    
    def __init__(self, 
                 depth_range: Tuple[float, float] = (2850.0, 2950.0),  # 调整为100ft范围
                 azimuth_sectors: int = 8,
                 batch_size: int = 50):
        """
        初始化增量特征工程器
        
        Args:
            depth_range: 深度范围 (ft)
            azimuth_sectors: 方位角扇区数
            batch_size: 批处理大小
        """
        self.depth_range = depth_range
        self.azimuth_sectors = azimuth_sectors
        self.batch_size = batch_size
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        
        # 方位角扇区划分 (360度/8扇区 = 45度每扇区)
        self.sector_size = 360 // self.azimuth_sectors
        
        logger.info(f"初始化增量特征工程器:")
        logger.info(f"  深度范围: {depth_range[0]:.1f}-{depth_range[1]:.1f} ft")
        logger.info(f"  方位角扇区: {azimuth_sectors} 个")
        logger.info(f"  批处理大小: {batch_size}")
    
    def estimate_total_samples(self) -> int:
        """
        估算总样本数，用于初始化HDF5数据集
        """
        logger.info("估算总样本数...")
        
        # 加载CAST数据获取深度信息
        cast_data = self.data_loader.load_cast_data()
        depths = cast_data['Depth']
        
        # 筛选深度范围
        depth_mask = (depths >= self.depth_range[0]) & (depths <= self.depth_range[1])
        n_depths = np.sum(depth_mask)
        
        # 估算总样本数: 深度点数 × 接收器数 × 方位角扇区数
        n_receivers = 13  # XSILMR有13个接收器
        total_samples = n_depths * n_receivers * self.azimuth_sectors
        
        logger.info(f"估算结果:")
        logger.info(f"  筛选深度点数: {n_depths}")
        logger.info(f"  接收器数: {n_receivers}")
        logger.info(f"  方位角扇区数: {self.azimuth_sectors}")
        logger.info(f"  预估总样本数: {total_samples}")
        
        return total_samples
    
    def generate_features_to_hdf5(self, hdf5_path: str = "data/processed/features.h5") -> int:
        """
        生成特征并增量存储到HDF5文件
        
        Args:
            hdf5_path: HDF5文件路径
            
        Returns:
            实际生成的样本数
        """
        logger.info("开始增量特征工程处理...")
        
        # 1. 估算总样本数
        total_samples = self.estimate_total_samples()
        
        # 2. 创建HDF5管理器和批处理器
        hdf5_manager = HDF5DataManager(hdf5_path, mode='w')
        hdf5_manager.create_dataset_structure(
            total_samples=total_samples,
            image_shape=(127, 1024),
            vector_dim=8,
            chunk_size=100
        )
        
        batch_processor = BatchProcessor(hdf5_manager, batch_size=self.batch_size)
        
        try:
            # 3. 加载基础数据
            logger.info("加载基础数据...")
            cast_data = self.data_loader.load_cast_data()
            xsilmr_data = self.data_loader.load_xsilmr_data()
            
            # 4. 筛选深度范围
            filtered_cast, filtered_xsilmr = self.data_loader.filter_depth_range(
                min_depth=self.depth_range[0],
                max_depth=self.depth_range[1]
            )
            
            # 5. 计算绝对深度
            filtered_xsilmr = self.data_loader.calculate_absolute_depths(filtered_xsilmr)
            
            logger.info(f"筛选后数据:")
            logger.info(f"  CAST深度点数: {len(filtered_cast['Depth'])}")
            logger.info(f"  XSILMR深度点数: {len(filtered_xsilmr[7]['Depth'])}")
            
            # 6. 开始分块处理
            sample_id = 0
            
            # 按接收器处理
            for receiver_id in sorted(filtered_xsilmr.keys()):
                logger.info(f"处理接收器 {receiver_id}...")
                receiver_data = filtered_xsilmr[receiver_id]
                
                # 按方位角扇区处理
                for sector_idx in range(self.azimuth_sectors):
                    logger.info(f"  处理方位角扇区 {sector_idx + 1}/{self.azimuth_sectors}")
                    
                    # 获取扇区内的方位角数据
                    sector_cast_data = self._get_sector_cast_data(
                        filtered_cast, sector_idx
                    )
                    
                    # 获取XSILMR方位角数据
                    side_key = self._get_side_key(sector_idx)
                    if side_key not in receiver_data:
                        logger.warning(f"接收器 {receiver_id} 缺少 {side_key} 数据")
                        continue
                    
                    xsilmr_waves = receiver_data[side_key]
                    depths = receiver_data['AbsoluteDepth']
                    
                    # 逐深度点处理
                    self._process_depth_points(
                        xsilmr_waves, depths, sector_cast_data,
                        receiver_id, sector_idx, sample_id,
                        batch_processor
                    )
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                sample_id += len(depths) if 'depths' in locals() else 0
            
            # 7. 完成处理
            actual_samples = batch_processor.finalize()
            
            # 8. 导出摘要
            hdf5_manager.mode = 'r'  # 切换到读模式
            hdf5_manager.export_summary()
            
            logger.info(f"增量特征工程完成!")
            logger.info(f"实际生成样本数: {actual_samples}")
            logger.info(f"HDF5文件: {hdf5_path}")
            
            return actual_samples
            
        except Exception as e:
            logger.error(f"特征工程过程中发生错误: {e}")
            raise
        finally:
            # 清理资源
            del batch_processor
            del hdf5_manager
            gc.collect()
    
    def _get_sector_cast_data(self, cast_data: Dict, sector_idx: int) -> np.ndarray:
        """
        获取指定扇区的CAST数据
        
        Args:
            cast_data: CAST数据字典
            sector_idx: 扇区索引 (0-7)
            
        Returns:
            扇区CAST数据 (n_depths,)
        """
        zc_data = cast_data['Zc']  # 形状: (180, n_depths)
        
        # 计算扇区的方位角范围
        start_angle = sector_idx * self.sector_size
        end_angle = (sector_idx + 1) * self.sector_size
        
        # 获取扇区内的方位角索引 (每2度一个方位角)
        start_idx = start_angle // 2
        end_idx = end_angle // 2
        
        # 计算扇区内的平均窜槽比例
        sector_zc = zc_data[start_idx:end_idx, :]  # (sector_azimuths, n_depths)
        
        # 计算每个深度点的窜槽比例 (Zc < 2.5 为窜槽)
        channeling_mask = sector_zc < 2.5
        channeling_ratios = np.mean(channeling_mask, axis=0)  # (n_depths,)
        
        return channeling_ratios
    
    def _get_side_key(self, sector_idx: int) -> str:
        """根据扇区索引获取对应的方位键"""
        sides = ['SideA', 'SideB', 'SideC', 'SideD', 
                'SideE', 'SideF', 'SideG', 'SideH']
        return sides[sector_idx]
    
    def _process_depth_points(self, 
                            xsilmr_waves: np.ndarray,
                            depths: np.ndarray,
                            cast_ratios: np.ndarray,
                            receiver_id: int,
                            sector_idx: int,
                            base_sample_id: int,
                            batch_processor: BatchProcessor):
        """
        处理单个接收器-方位角组合的所有深度点
        
        Args:
            xsilmr_waves: XSILMR波形数据 (1024, n_depths)
            depths: 深度数组
            cast_ratios: CAST窜槽比例数组
            receiver_id: 接收器ID
            sector_idx: 扇区索引
            base_sample_id: 基础样本ID
            batch_processor: 批处理器
        """
        n_time, n_depths = xsilmr_waves.shape
        
        for depth_idx in range(n_depths):
            if depth_idx % 100 == 0:
                logger.debug(f"    处理深度点 {depth_idx}/{n_depths}")
            
            # 提取单个深度点的波形
            waveform = xsilmr_waves[:, depth_idx]  # (1024,)
            
            try:
                # 1. 信号处理
                filtered_waveform = self.signal_processor.apply_highpass_filter(waveform)
                
                # 2. 生成尺度图 (图像特征)
                scalogram = self.signal_processor.generate_scalogram(filtered_waveform)
                
                # 3. 提取物理特征 (数值特征)
                physical_features = self.signal_processor.extract_physical_features(filtered_waveform)
                
                # 转换为特征向量
                vector_features = np.array([
                    physical_features['max_amplitude'],
                    physical_features['rms_amplitude'],
                    physical_features['energy'],
                    physical_features['zero_crossings'],
                    physical_features['dominant_frequency'],
                    physical_features['spectral_centroid'],
                    receiver_id,  # 接收器ID作为特征
                    sector_idx    # 方位角扇区作为特征
                ], dtype=np.float32)
                
                # 4. 获取标签 (窜槽比例)
                label = cast_ratios[depth_idx]
                
                # 5. 元数据
                metadata = (
                    depths[depth_idx],  # depth
                    receiver_id,        # receiver_id
                    sector_idx,         # azimuth_sector
                    base_sample_id + depth_idx  # sample_id
                )
                
                # 6. 添加到批处理器
                batch_processor.add_sample(
                    image_feature=scalogram,
                    vector_feature=vector_features,
                    label=label,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.warning(f"处理深度点 {depth_idx} 时出错: {e}")
                continue


class FeatureEngineer:
    """
    传统特征工程器 (兼容性保持)
    注意: 此类保留是为了兼容性，建议使用 IncrementalFeatureEngineer
    """
    
    def __init__(self, depth_range: Tuple[float, float] = (2850.0, 2950.0)):
        """
        初始化特征工程器
        
        Args:
            depth_range: 分析深度范围 (ft) - 调整为较小范围
        """
        self.depth_range = depth_range
        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        
        logger.warning("使用传统FeatureEngineer可能导致内存不足")
        logger.warning("建议使用IncrementalFeatureEngineer进行大规模数据处理")
        
    def generate_training_data(self, target_receiver: int = 7, 
                             max_samples: int = 200) -> Tuple:
        """
        生成训练数据 (限制样本数以控制内存使用)
        
        Args:
            target_receiver: 目标接收器
            max_samples: 最大样本数
            
        Returns:
            图像特征, 数值特征, 标签
        """
        logger.info("生成训练数据 (传统方法，限制样本数)...")
        logger.warning(f"样本数限制为 {max_samples} 以控制内存使用")
        
        # 加载数据
        cast_data = self.data_loader.load_cast_data()
        xsilmr_data = self.data_loader.load_xsilmr_data()
        
        # 筛选深度范围
        filtered_cast, filtered_xsilmr = self.data_loader.filter_depth_range(
            min_depth=self.depth_range[0],
            max_depth=self.depth_range[1]
        )
        
        # 计算绝对深度
        filtered_xsilmr = self.data_loader.calculate_absolute_depths(filtered_xsilmr)
        
        if target_receiver not in filtered_xsilmr:
            raise ValueError(f"目标接收器 {target_receiver} 不存在")
        
        # 获取目标接收器数据
        receiver_data = filtered_xsilmr[target_receiver]
        xsilmr_depths = receiver_data['AbsoluteDepth']
        cast_depths = filtered_cast['Depth']
        cast_zc = filtered_cast['Zc']
        
        # 准备存储
        image_features = []
        vector_features = []
        labels = []
        
        sample_count = 0
        
        # 处理方位A的数据
        if 'SideA' in receiver_data:
            waveforms = receiver_data['SideA']  # (1024, n_depths)
            
            # 创建方位角窜槽标签
            azimuth_range = (0, 45)  # 方位A对应0-45度
            sector_labels = self._create_azimuth_labels(
                cast_depths, cast_zc, xsilmr_depths, azimuth_range
            )
            
            # 处理波形数据
            n_depths = min(waveforms.shape[1], len(sector_labels), max_samples)
            
            for i in range(n_depths):
                try:
                    waveform = waveforms[:, i]
                    
                    # 应用高通滤波
                    filtered_waveform = self.signal_processor.apply_highpass_filter(waveform)
                    
                    # 生成尺度图
                    scalogram = self.signal_processor.generate_scalogram(filtered_waveform)
                    
                    # 提取物理特征
                    features = self.signal_processor.extract_physical_features(filtered_waveform)
                    
                    # 创建特征向量
                    feature_vector = np.array([
                        features['max_amplitude'],
                        features['rms_amplitude'],
                        features['energy'],
                        features['zero_crossings'],
                        features['dominant_frequency'],
                        features['spectral_centroid'],
                        target_receiver,
                        0  # 方位A的索引
                    ])
                    
                    image_features.append(scalogram)
                    vector_features.append(feature_vector)
                    labels.append(sector_labels[i])
                    
                    sample_count += 1
                    
                    if sample_count >= max_samples:
                        break
                        
                except Exception as e:
                    logger.warning(f"处理样本 {i} 时出错: {e}")
                    continue
        
        logger.info(f"传统特征工程完成: 生成 {len(image_features)} 个样本")
        
        return np.array(image_features), np.array(vector_features), np.array(labels)
    
    def _create_azimuth_labels(self, cast_depths: np.ndarray, cast_zc: np.ndarray,
                             xsilmr_depths: np.ndarray, azimuth_range: Tuple[int, int]) -> np.ndarray:
        """创建方位角标签"""
        start_angle, end_angle = azimuth_range
        
        # 方位角索引 (每2度一个)
        start_idx = start_angle // 2
        end_idx = end_angle // 2
        
        # 获取方位角范围内的数据
        azimuth_zc = cast_zc[start_idx:end_idx, :]  # (n_azimuths, n_depths)
        
        # 计算窜槽比例
        channeling_mask = azimuth_zc < 2.5
        channeling_ratios = np.mean(channeling_mask, axis=0)  # (n_depths,)
        
        # 插值到XSILMR深度
        interpolated_ratios = np.interp(xsilmr_depths, cast_depths, channeling_ratios)
        
        return interpolated_ratios 