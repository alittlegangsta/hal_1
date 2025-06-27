#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5数据管理器 - 用于大规模测井数据的增量存储和高效读取
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class HDF5DataManager:
    """HDF5数据管理器 - 支持增量写入和高效读取"""
    
    def __init__(self, file_path: str, mode: str = 'w'):
        """
        初始化HDF5数据管理器
        
        Args:
            file_path: HDF5文件路径
            mode: 文件打开模式 ('w': 写, 'r': 读, 'a': 追加)
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.file_handle = None
        
        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
    @contextmanager
    def open_file(self):
        """上下文管理器，安全地打开和关闭HDF5文件"""
        try:
            self.file_handle = h5py.File(self.file_path, self.mode)
            yield self.file_handle
        finally:
            if self.file_handle is not None:
                self.file_handle.close()
                self.file_handle = None
    
    def create_dataset_structure(self, 
                               total_samples: int,
                               image_shape: Tuple[int, int] = (127, 1024),
                               vector_dim: int = 8,
                               chunk_size: int = 100):
        """
        创建HDF5数据集结构
        
        Args:
            total_samples: 预期的总样本数
            image_shape: 图像特征形状
            vector_dim: 数值特征维度
            chunk_size: 数据块大小，用于优化I/O性能
        """
        logger.info(f"创建HDF5数据集结构: {self.file_path}")
        logger.info(f"预期样本数: {total_samples}, 图像形状: {image_shape}, 向量维度: {vector_dim}")
        
        # 调整chunk_size以确保不超过总样本数
        actual_chunk_size = min(chunk_size, total_samples)
        if actual_chunk_size != chunk_size:
            logger.info(f"调整chunk_size: {chunk_size} -> {actual_chunk_size} (不能超过总样本数)")
        
        with self.open_file() as f:
            # 创建数据集 - 使用gzip压缩节省空间
            
            # 图像特征数据集 (scalograms)
            f.create_dataset(
                'image_features',
                shape=(total_samples, *image_shape),
                dtype=np.float32,
                chunks=(actual_chunk_size, *image_shape),
                compression='gzip',
                compression_opts=6,
                shuffle=True  # 重排数据以提高压缩率
            )
            
            # 数值特征数据集
            f.create_dataset(
                'vector_features',
                shape=(total_samples, vector_dim),
                dtype=np.float32,
                chunks=(actual_chunk_size, vector_dim),
                compression='gzip',
                compression_opts=6,
                shuffle=True
            )
            
            # 标签数据集
            f.create_dataset(
                'labels',
                shape=(total_samples,),
                dtype=np.float32,
                chunks=(actual_chunk_size,),
                compression='gzip',
                compression_opts=6
            )
            
            # 元数据数据集 - 存储深度、接收器、方位角等信息
            f.create_dataset(
                'metadata',
                shape=(total_samples,),
                dtype=[
                    ('depth', 'f4'),
                    ('receiver_id', 'i4'),
                    ('azimuth_sector', 'i4'),
                    ('sample_id', 'i4')
                ],
                chunks=(actual_chunk_size,),
                compression='gzip',
                compression_opts=6
            )
            
            # 添加属性信息
            f.attrs['total_samples'] = total_samples
            f.attrs['image_shape'] = image_shape
            f.attrs['vector_dim'] = vector_dim
            f.attrs['chunk_size'] = actual_chunk_size
            f.attrs['created_by'] = 'HAL Well Logging Analysis'
            
        logger.info("HDF5数据集结构创建完成")
    
    def write_batch(self, 
                   start_idx: int,
                   image_features: np.ndarray,
                   vector_features: np.ndarray,
                   labels: np.ndarray,
                   metadata: Optional[np.ndarray] = None):
        """
        批量写入数据到HDF5文件
        
        Args:
            start_idx: 起始索引
            image_features: 图像特征数组
            vector_features: 数值特征数组
            labels: 标签数组
            metadata: 元数据数组 (可选)
        """
        batch_size = len(image_features)
        end_idx = start_idx + batch_size
        
        logger.debug(f"写入批次 [{start_idx}:{end_idx}], 大小: {batch_size}")
        
        # 确保文件是以写入模式打开
        current_mode = self.mode
        self.mode = 'r+' if current_mode == 'r' else 'a'
        
        try:
            with self.open_file() as f:
                # 写入图像特征
                f['image_features'][start_idx:end_idx] = image_features.astype(np.float32)
                
                # 写入数值特征
                f['vector_features'][start_idx:end_idx] = vector_features.astype(np.float32)
                
                # 写入标签
                f['labels'][start_idx:end_idx] = labels.astype(np.float32)
                
                # 写入元数据 (如果提供)
                if metadata is not None:
                    f['metadata'][start_idx:end_idx] = metadata
                
                # 刷新缓冲区，确保数据写入磁盘
                f.flush()
        except Exception as e:
            logger.error(f"写入批次数据时出错: {e}")
            raise
        finally:
            # 恢复原始模式
            self.mode = current_mode
    
    def read_batch(self, start_idx: int, batch_size: int) -> Dict[str, np.ndarray]:
        """
        从HDF5文件读取批次数据
        
        Args:
            start_idx: 起始索引
            batch_size: 批次大小
            
        Returns:
            包含各类特征数据的字典
        """
        end_idx = start_idx + batch_size
        
        with self.open_file() as f:
            data = {
                'image_features': f['image_features'][start_idx:end_idx],
                'vector_features': f['vector_features'][start_idx:end_idx],
                'labels': f['labels'][start_idx:end_idx]
            }
            
            # 读取元数据 (如果存在)
            if 'metadata' in f:
                data['metadata'] = f['metadata'][start_idx:end_idx]
                
        return data
    
    def read_all_data(self) -> Dict[str, np.ndarray]:
        """
        读取所有数据 (小心使用，可能消耗大量内存)
        
        Returns:
            包含所有数据的字典
        """
        logger.warning("读取所有数据到内存，请确保有足够的内存空间")
        
        with self.open_file() as f:
            data = {
                'image_features': f['image_features'][:],
                'vector_features': f['vector_features'][:],
                'labels': f['labels'][:]
            }
            
            if 'metadata' in f:
                data['metadata'] = f['metadata'][:]
                
        logger.info(f"已读取 {len(data['labels'])} 个样本到内存")
        return data
    
    def get_data_info(self) -> Dict:
        """获取数据集信息"""
        with self.open_file() as f:
            info = {
                'total_samples': f.attrs.get('total_samples', len(f['labels'])),
                'image_shape': f.attrs.get('image_shape', f['image_features'].shape[1:]),
                'vector_dim': f.attrs.get('vector_dim', f['vector_features'].shape[1]),
                'chunk_size': f.attrs.get('chunk_size', 100),
                'file_size_mb': self.file_path.stat().st_size / (1024 * 1024),
                'datasets': list(f.keys())
            }
            
            # 检查每个数据集的实际大小
            for dataset_name in ['image_features', 'vector_features', 'labels']:
                if dataset_name in f:
                    dataset = f[dataset_name]
                    info[f'{dataset_name}_shape'] = dataset.shape
                    info[f'{dataset_name}_dtype'] = str(dataset.dtype)
                    
        return info
    
    def create_data_iterator(self, batch_size: int = 32, shuffle: bool = False):
        """
        创建数据迭代器，用于模型训练
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据顺序
            
        Yields:
            批次数据字典
        """
        with self.open_file() as f:
            total_samples = len(f['labels'])
            
            # 创建索引数组
            indices = np.arange(total_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            # 分批迭代
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_indices = indices[start:end]
                
                # 读取批次数据
                batch_data = {
                    'image_features': f['image_features'][batch_indices],
                    'vector_features': f['vector_features'][batch_indices],
                    'labels': f['labels'][batch_indices]
                }
                
                if 'metadata' in f:
                    batch_data['metadata'] = f['metadata'][batch_indices]
                    
                yield batch_data
    
    def split_train_test(self, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[int], List[int]]:
        """
        生成训练和测试索引
        
        Args:
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            (train_indices, test_indices)
        """
        with self.open_file() as f:
            total_samples = len(f['labels'])
            
        # 生成随机索引
        np.random.seed(random_seed)
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        
        # 划分训练和测试集
        n_test = int(total_samples * test_ratio)
        test_indices = all_indices[:n_test].tolist()
        train_indices = all_indices[n_test:].tolist()
        
        logger.info(f"数据划分: 训练集={len(train_indices)}, 测试集={len(test_indices)}")
        
        return train_indices, test_indices
    
    def export_summary(self, output_path: str = None):
        """导出数据集摘要信息"""
        info = self.get_data_info()
        
        if output_path is None:
            output_path = str(self.file_path.with_suffix('.summary.txt'))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("HDF5数据集摘要信息\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"文件路径: {self.file_path}\n")
            f.write(f"文件大小: {info['file_size_mb']:.2f} MB\n")
            f.write(f"总样本数: {info['total_samples']}\n")
            f.write(f"图像特征形状: {info['image_shape']}\n")
            f.write(f"数值特征维度: {info['vector_dim']}\n")
            f.write(f"数据块大小: {info['chunk_size']}\n\n")
            
            f.write("数据集详情:\n")
            f.write("-" * 20 + "\n")
            for key, value in info.items():
                if key.endswith('_shape') or key.endswith('_dtype'):
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"数据集摘要已导出: {output_path}")


class BatchProcessor:
    """批处理器 - 管理分块特征工程流程"""
    
    def __init__(self, hdf5_manager: HDF5DataManager, batch_size: int = 50):
        """
        初始化批处理器
        
        Args:
            hdf5_manager: HDF5数据管理器
            batch_size: 批处理大小
        """
        self.hdf5_manager = hdf5_manager
        self.batch_size = batch_size
        self.current_index = 0
        
        # 批次缓存
        self.image_batch = []
        self.vector_batch = []
        self.label_batch = []
        self.metadata_batch = []
    
    def add_sample(self, 
                  image_feature: np.ndarray,
                  vector_feature: np.ndarray,
                  label: float,
                  metadata: Optional[Tuple] = None):
        """
        添加单个样本到批次缓存
        
        Args:
            image_feature: 图像特征
            vector_feature: 数值特征
            label: 标签
            metadata: 元数据元组 (depth, receiver_id, azimuth_sector, sample_id)
        """
        self.image_batch.append(image_feature)
        self.vector_batch.append(vector_feature)
        self.label_batch.append(label)
        
        if metadata is not None:
            self.metadata_batch.append(metadata)
        
        # 如果批次已满，写入HDF5文件
        if len(self.image_batch) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """将批次数据写入HDF5文件并清空缓存"""
        if len(self.image_batch) == 0:
            return
        
        # 转换为numpy数组
        image_features = np.array(self.image_batch)
        vector_features = np.array(self.vector_batch)
        labels = np.array(self.label_batch)
        
        metadata = None
        if self.metadata_batch:
            metadata = np.array(self.metadata_batch, dtype=[
                ('depth', 'f4'),
                ('receiver_id', 'i4'),
                ('azimuth_sector', 'i4'),
                ('sample_id', 'i4')
            ])
        
        # 写入HDF5文件
        self.hdf5_manager.write_batch(
            self.current_index,
            image_features,
            vector_features,
            labels,
            metadata
        )
        
        # 更新索引
        batch_size = len(self.image_batch)
        self.current_index += batch_size
        
        logger.info(f"已写入批次: {batch_size} 个样本, 累计: {self.current_index}")
        
        # 清空缓存，释放内存
        self.image_batch.clear()
        self.vector_batch.clear()
        self.label_batch.clear()
        self.metadata_batch.clear()
        
        # 强制垃圾回收
        import gc
        gc.collect()
    
    def finalize(self):
        """完成处理，写入剩余数据"""
        if len(self.image_batch) > 0:
            self.flush_batch()
        
        logger.info(f"批处理完成，总计处理: {self.current_index} 个样本")
        return self.current_index 