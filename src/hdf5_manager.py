"""
HDF5数据管理器 - 支持大规模数据的增量存储和高效读取
"""
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import gc
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class HDF5DataManager:
    """HDF5数据集管理器 - 支持大规模数据的增量存储"""
    
    def __init__(self, filepath: str, mode: str = 'w'):
        """
        初始化HDF5数据管理器
        
        Args:
            filepath: HDF5文件路径
            mode: 文件打开模式 ('w', 'r', 'a')
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self.file = None
        self.datasets = {}
        
        # 确保目录存在
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 打开文件
        self._open_file()
        
        logger.info(f"HDF5数据管理器初始化: {self.filepath}, 模式: {mode}")
    
    def _open_file(self):
        """打开HDF5文件"""
        try:
            self.file = h5py.File(str(self.filepath), self.mode)
            logger.debug(f"HDF5文件已打开: {self.filepath}")
        except Exception as e:
            logger.error(f"无法打开HDF5文件 {self.filepath}: {e}")
            raise
    
    def create_dataset_structure(self, 
                               total_samples: int,
                               image_shape: Tuple[int, int] = (127, 1024),
                               vector_dim: int = 8,
                               chunk_size: int = 100):
        """
        创建数据集结构
        
        Args:
            total_samples: 预估总样本数
            image_shape: 图像特征形状
            vector_dim: 向量特征维度
            chunk_size: 分块大小
        """
        if self.mode == 'r':
            raise ValueError("只读模式下无法创建数据集")
        
        logger.info(f"创建HDF5数据集结构:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  图像形状: {image_shape}")
        logger.info(f"  向量维度: {vector_dim}")
        logger.info(f"  分块大小: {chunk_size}")
        
        # 图像特征数据集
        self.datasets['images'] = self.file.create_dataset(
            'images',
            shape=(total_samples, *image_shape),
            dtype=np.float32,
            chunks=(chunk_size, *image_shape),
            compression='gzip',
            compression_opts=6,
            maxshape=(None, *image_shape)  # 允许动态扩展
        )
        
        # 向量特征数据集
        self.datasets['vectors'] = self.file.create_dataset(
            'vectors',
            shape=(total_samples, vector_dim),
            dtype=np.float32,
            chunks=(chunk_size, vector_dim),
            compression='gzip',
            compression_opts=6,
            maxshape=(None, vector_dim)
        )
        
        # 标签数据集
        self.datasets['labels'] = self.file.create_dataset(
            'labels',
            shape=(total_samples,),
            dtype=np.float32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=6,
            maxshape=(None,)
        )
        
        # 元数据数据集
        self.datasets['metadata'] = self.file.create_dataset(
            'metadata',
            shape=(total_samples, 4),  # depth, receiver_id, azimuth_sector, sample_id
            dtype=np.float32,
            chunks=(chunk_size, 4),
            compression='gzip',
            compression_opts=6,
            maxshape=(None, 4)
        )
        
        # 存储数据集描述信息
        self._save_dataset_info(total_samples, image_shape, vector_dim, chunk_size)
        
        logger.info("HDF5数据集结构创建完成")
    
    def _save_dataset_info(self, total_samples: int, image_shape: Tuple, 
                          vector_dim: int, chunk_size: int):
        """保存数据集描述信息"""
        info = {
            'total_samples': total_samples,
            'image_shape': image_shape,
            'vector_dim': vector_dim,
            'chunk_size': chunk_size,
            'created_time': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # 将信息存储为HDF5属性
        for key, value in info.items():
            if isinstance(value, (list, tuple)):
                self.file.attrs[key] = json.dumps(value)
            else:
                self.file.attrs[key] = str(value)
    
    def write_batch(self, 
                   start_idx: int,
                   images: np.ndarray,
                   vectors: np.ndarray,
                   labels: np.ndarray,
                   metadata: np.ndarray):
        """
        批量写入数据
        
        Args:
            start_idx: 起始索引
            images: 图像特征数组 (batch_size, height, width)
            vectors: 向量特征数组 (batch_size, vector_dim)
            labels: 标签数组 (batch_size,)
            metadata: 元数据数组 (batch_size, 4)
        """
        if self.mode == 'r':
            raise ValueError("只读模式下无法写入数据")
        
        batch_size = len(images)
        end_idx = start_idx + batch_size
        
        try:
            # 检查是否需要扩展数据集
            current_size = self.datasets['images'].shape[0]
            if end_idx > current_size:
                new_size = max(end_idx, current_size + batch_size)
                self._resize_datasets(new_size)
            
            # 写入数据
            self.datasets['images'][start_idx:end_idx] = images
            self.datasets['vectors'][start_idx:end_idx] = vectors
            self.datasets['labels'][start_idx:end_idx] = labels
            self.datasets['metadata'][start_idx:end_idx] = metadata
            
            # 强制写入磁盘
            self.file.flush()
            
            logger.debug(f"批量数据写入完成: 索引 {start_idx}-{end_idx-1}")
            
        except Exception as e:
            logger.error(f"批量写入数据失败: {e}")
            raise
    
    def _resize_datasets(self, new_size: int):
        """扩展数据集大小"""
        logger.info(f"扩展数据集大小到: {new_size}")
        
        for name, dataset in self.datasets.items():
            if name == 'metadata':
                dataset.resize((new_size, 4))
            elif name in ['labels']:
                dataset.resize((new_size,))
            elif name == 'vectors':
                current_vector_dim = dataset.shape[1]
                dataset.resize((new_size, current_vector_dim))
            elif name == 'images':
                current_shape = dataset.shape[1:]
                dataset.resize((new_size, *current_shape))
    
    def read_batch(self, start_idx: int, batch_size: int) -> Tuple:
        """
        批量读取数据
        
        Args:
            start_idx: 起始索引
            batch_size: 批大小
            
        Returns:
            图像特征, 向量特征, 标签, 元数据
        """
        if self.mode == 'w':
            logger.warning("写模式下读取数据，切换到追加模式")
            self.file.close()
            self.mode = 'a'
            self._open_file()
            self._load_existing_datasets()
        
        end_idx = min(start_idx + batch_size, self.get_total_samples())
        
        images = self.datasets['images'][start_idx:end_idx]
        vectors = self.datasets['vectors'][start_idx:end_idx]
        labels = self.datasets['labels'][start_idx:end_idx]
        metadata = self.datasets['metadata'][start_idx:end_idx]
        
        return images, vectors, labels, metadata
    
    def _load_existing_datasets(self):
        """加载现有数据集"""
        self.datasets = {
            'images': self.file['images'],
            'vectors': self.file['vectors'],
            'labels': self.file['labels'],
            'metadata': self.file['metadata']
        }
    
    def get_total_samples(self) -> int:
        """获取总样本数"""
        if 'labels' in self.datasets:
            return self.datasets['labels'].shape[0]
        elif 'labels' in self.file:
            return self.file['labels'].shape[0]
        else:
            return 0
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        info = {}
        for key in self.file.attrs.keys():
            value = self.file.attrs[key]
            if isinstance(value, str) and (value.startswith('[') or value.startswith('(')):
                try:
                    info[key] = json.loads(value)
                except:
                    info[key] = value
            else:
                info[key] = value
        return info
    
    def export_summary(self, output_path: Optional[str] = None):
        """导出数据集摘要"""
        if output_path is None:
            output_path = str(self.filepath.parent / f"{self.filepath.stem}_summary.txt")
        
        info = self.get_dataset_info()
        total_samples = self.get_total_samples()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HDF5数据集摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"文件路径: {self.filepath}\n")
            f.write(f"文件大小: {self.filepath.stat().st_size / (1024**3):.2f} GB\n")
            f.write(f"实际样本数: {total_samples}\n\n")
            
            f.write("数据集信息:\n")
            for key, value in info.items():
                f.write(f"  {key}: {value}\n")
            
            if total_samples > 0:
                f.write("\n数据统计:\n")
                
                # 标签统计
                labels = self.datasets['labels'][:total_samples]
                f.write(f"  标签范围: {np.min(labels):.4f} - {np.max(labels):.4f}\n")
                f.write(f"  标签均值: {np.mean(labels):.4f}\n")
                f.write(f"  标签标准差: {np.std(labels):.4f}\n")
                
                # 高窜槽样本统计
                high_channeling = np.sum(labels > 0.3)
                f.write(f"  高窜槽样本 (>0.3): {high_channeling} ({high_channeling/total_samples*100:.1f}%)\n")
        
        logger.info(f"数据集摘要已导出: {output_path}")
    
    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            logger.debug("HDF5文件已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BatchProcessor:
    """批处理器 - 支持增量数据写入"""
    
    def __init__(self, hdf5_manager: HDF5DataManager, batch_size: int = 100):
        """
        初始化批处理器
        
        Args:
            hdf5_manager: HDF5管理器
            batch_size: 批处理大小
        """
        self.hdf5_manager = hdf5_manager
        self.batch_size = batch_size
        
        # 批处理缓冲区
        self.buffer_images = []
        self.buffer_vectors = []
        self.buffer_labels = []
        self.buffer_metadata = []
        
        # 写入统计
        self.total_written = 0
        self.current_batch_idx = 0
        
        logger.info(f"批处理器初始化: 批大小 = {batch_size}")
    
    def add_sample(self, 
                  image_feature: np.ndarray,
                  vector_feature: np.ndarray,
                  label: float,
                  metadata: Tuple):
        """
        添加单个样本到批处理缓冲区
        
        Args:
            image_feature: 图像特征 (height, width)
            vector_feature: 向量特征 (vector_dim,)
            label: 标签值
            metadata: 元数据 (depth, receiver_id, azimuth_sector, sample_id)
        """
        self.buffer_images.append(image_feature)
        self.buffer_vectors.append(vector_feature)
        self.buffer_labels.append(label)
        self.buffer_metadata.append(metadata)
        
        # 检查是否需要写入
        if len(self.buffer_images) >= self.batch_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """将缓冲区数据写入HDF5文件"""
        if not self.buffer_images:
            return
        
        batch_size = len(self.buffer_images)
        
        # 转换为numpy数组
        images_array = np.array(self.buffer_images, dtype=np.float32)
        vectors_array = np.array(self.buffer_vectors, dtype=np.float32)
        labels_array = np.array(self.buffer_labels, dtype=np.float32)
        metadata_array = np.array(self.buffer_metadata, dtype=np.float32)
        
        # 写入HDF5
        start_idx = self.total_written
        self.hdf5_manager.write_batch(
            start_idx=start_idx,
            images=images_array,
            vectors=vectors_array,
            labels=labels_array,
            metadata=metadata_array
        )
        
        # 更新统计
        self.total_written += batch_size
        self.current_batch_idx += 1
        
        logger.info(f"批次 {self.current_batch_idx} 已写入: {batch_size} 个样本 "
                   f"(总计: {self.total_written})")
        
        # 清空缓冲区并强制垃圾回收
        self.buffer_images.clear()
        self.buffer_vectors.clear()
        self.buffer_labels.clear()
        self.buffer_metadata.clear()
        gc.collect()
    
    def finalize(self) -> int:
        """
        完成处理，写入剩余数据
        
        Returns:
            总写入样本数
        """
        # 写入剩余数据
        if self.buffer_images:
            self._flush_buffer()
        
        logger.info(f"批处理完成: 总计写入 {self.total_written} 个样本")
        return self.total_written


class HDF5DataLoader:
    """HDF5数据加载器 - 用于模型训练时的高效数据读取"""
    
    def __init__(self, hdf5_path: str):
        """
        初始化数据加载器
        
        Args:
            hdf5_path: HDF5文件路径
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {hdf5_path}")
        
        self.manager = HDF5DataManager(str(hdf5_path), mode='r')
        self.manager._load_existing_datasets()
        
        logger.info(f"HDF5数据加载器初始化: {hdf5_path}")
    
    def get_batch_generator(self, batch_size: int = 32, shuffle: bool = True):
        """
        获取批数据生成器
        
        Args:
            batch_size: 批大小
            shuffle: 是否打乱数据
            
        Yields:
            图像特征, 向量特征, 标签
        """
        total_samples = self.manager.get_total_samples()
        indices = np.arange(total_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # 读取批数据
            images = self.manager.datasets['images'][batch_indices]
            vectors = self.manager.datasets['vectors'][batch_indices]
            labels = self.manager.datasets['labels'][batch_indices]
            
            yield images, vectors, labels
    
    def load_all_data(self) -> Tuple:
        """
        加载所有数据到内存 (仅用于小数据集)
        
        Returns:
            图像特征, 向量特征, 标签
        """
        total_samples = self.manager.get_total_samples()
        logger.warning(f"将 {total_samples} 个样本全部加载到内存")
        
        images = self.manager.datasets['images'][:]
        vectors = self.manager.datasets['vectors'][:]
        labels = self.manager.datasets['labels'][:]
        
        return images, vectors, labels
    
    def get_sample(self, index: int) -> Tuple:
        """
        获取单个样本
        
        Args:
            index: 样本索引
            
        Returns:
            图像特征, 向量特征, 标签, 元数据
        """
        image = self.manager.datasets['images'][index]
        vector = self.manager.datasets['vectors'][index]
        label = self.manager.datasets['labels'][index]
        metadata = self.manager.datasets['metadata'][index]
        
        return image, vector, label, metadata
    
    def close(self):
        """关闭数据加载器"""
        self.manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 