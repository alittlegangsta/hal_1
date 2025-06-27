#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5增量处理简化演示 - 解决核心问题
"""

import sys
import numpy as np
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append('src')

from src.hdf5_manager import HDF5DataManager, BatchProcessor
from src.data_loader import DataLoader
from src.signal_processing import SignalProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hdf5_basic_functionality():
    """测试HDF5基本功能"""
    logger.info("🧪 测试HDF5基本功能...")
    
    test_path = "data/processed/test_features.h5"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # 删除旧文件（如果存在）
    if Path(test_path).exists():
        Path(test_path).unlink()
    
    try:
        # 1. 创建HDF5管理器
        manager = HDF5DataManager(test_path, mode='w')
        
        # 2. 创建数据集结构
        total_samples = 100
        manager.create_dataset_structure(
            total_samples=total_samples,
            image_shape=(127, 1024),
            vector_dim=8,
            chunk_size=50
        )
        
        # 3. 创建批处理器
        batch_processor = BatchProcessor(manager, batch_size=20)
        
        # 4. 生成测试数据并写入
        logger.info("生成测试数据...")
        for i in range(total_samples):
            # 生成模拟数据
            image_feature = np.random.random((127, 1024)).astype(np.float32)
            vector_feature = np.random.random(8).astype(np.float32)
            label = np.random.random()
            metadata = (2850.0 + i, 7, 0, i)  # (depth, receiver_id, sector, sample_id)
            
            batch_processor.add_sample(image_feature, vector_feature, label, metadata)
            
            if (i + 1) % 20 == 0:
                logger.info(f"已添加 {i + 1} 个样本")
        
        # 5. 完成处理
        actual_samples = batch_processor.finalize()
        logger.info(f"✅ 成功写入 {actual_samples} 个样本")
        
        # 6. 测试读取
        manager.mode = 'r'
        info = manager.get_data_info()
        logger.info(f"HDF5文件信息: {info}")
        
        # 读取部分数据
        sample_data = manager.read_batch(0, 10)
        logger.info(f"读取测试: 图像={sample_data['image_features'].shape}, 向量={sample_data['vector_features'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"HDF5测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        if Path(test_path).exists():
            Path(test_path).unlink()
            logger.info("已清理测试文件")

def test_real_data_processing():
    """测试真实数据处理（小规模）"""
    logger.info("🔬 测试真实数据处理...")
    
    try:
        # 1. 加载真实数据
        data_loader = DataLoader()
        signal_processor = SignalProcessor()
        
        logger.info("加载CAST和XSILMR数据...")
        cast_data = data_loader.load_cast_data()
        xsilmr_data = data_loader.load_xsilmr_data()
        
        # 2. 筛选小范围数据 (仅10ft范围测试)
        test_depth_range = (2850.0, 2860.0)
        filtered_cast, filtered_xsilmr = data_loader.filter_depth_range(
            min_depth=test_depth_range[0],
            max_depth=test_depth_range[1]
        )
        
        # 3. 计算绝对深度
        filtered_xsilmr = data_loader.calculate_absolute_depths(filtered_xsilmr)
        
        logger.info(f"筛选后深度点数: CAST={len(filtered_cast['Depth'])}, XSILMR={len(filtered_xsilmr[7]['Depth'])}")
        
        # 4. 创建HDF5文件
        test_path = "data/processed/real_test_features.h5"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        if Path(test_path).exists():
            Path(test_path).unlink()
        
        # 估算样本数（只处理一个接收器的一个方位）
        n_depths = len(filtered_xsilmr[7]['Depth'])
        total_samples = n_depths  # 只处理接收器7的方位A
        
        manager = HDF5DataManager(test_path, mode='w')
        manager.create_dataset_structure(
            total_samples=total_samples,
            image_shape=(127, 1024),
            vector_dim=8,
            chunk_size=50
        )
        
        batch_processor = BatchProcessor(manager, batch_size=20)
        
        # 5. 处理数据
        receiver_id = 7
        if receiver_id in filtered_xsilmr and 'SideA' in filtered_xsilmr[receiver_id]:
            logger.info(f"处理接收器 {receiver_id} 的方位A数据...")
            
            waveforms = filtered_xsilmr[receiver_id]['SideA']  # (1024, n_depths)
            depths = filtered_xsilmr[receiver_id]['AbsoluteDepth']
            
            # 创建方位角窜槽标签
            cast_depths = filtered_cast['Depth']
            cast_zc = filtered_cast['Zc']
            azimuth_range = (0, 45)  # 方位A对应0-45度
            
            # 方位角索引
            start_idx = azimuth_range[0] // 2
            end_idx = azimuth_range[1] // 2
            azimuth_zc = cast_zc[start_idx:end_idx, :]
            
            # 计算窜槽比例
            channeling_mask = azimuth_zc < 2.5
            channeling_ratios = np.mean(channeling_mask, axis=0)
            
            # 插值到XSILMR深度
            sector_labels = np.interp(depths, cast_depths, channeling_ratios)
            
            # 处理每个深度点
            processed_count = 0
            for depth_idx in range(min(len(depths), len(sector_labels))):
                try:
                    # 提取波形
                    waveform = waveforms[:, depth_idx]  # (1024,)
                    
                    # 信号处理
                    filtered_waveform = signal_processor.apply_highpass_filter(waveform)
                    scalogram = signal_processor.generate_scalogram(filtered_waveform)
                    physical_features = signal_processor.extract_physical_features(filtered_waveform)
                    
                    # 创建特征向量
                    vector_features = np.array([
                        physical_features['max_amplitude'],
                        physical_features['rms_amplitude'],
                        physical_features['energy'],
                        physical_features['zero_crossings'],
                        physical_features['dominant_frequency'],
                        physical_features['spectral_centroid'],
                        receiver_id,
                        0  # 方位A的索引
                    ], dtype=np.float32)
                    
                    # 标签和元数据
                    label = sector_labels[depth_idx]
                    metadata = (depths[depth_idx], receiver_id, 0, depth_idx)
                    
                    # 添加到批处理器
                    batch_processor.add_sample(scalogram, vector_features, label, metadata)
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"已处理 {processed_count} 个深度点")
                        
                except Exception as e:
                    logger.warning(f"处理深度点 {depth_idx} 时出错: {e}")
                    continue
            
            # 完成处理
            actual_samples = batch_processor.finalize()
            logger.info(f"✅ 成功处理 {actual_samples} 个真实样本")
            
            # 测试读取
            manager.mode = 'r'
            info = manager.get_data_info()
            logger.info(f"真实数据HDF5文件信息: 大小={info['file_size_mb']:.2f}MB, 样本数={info['total_samples']}")
            
            return True
        else:
            logger.error(f"接收器 {receiver_id} 的方位A数据不存在")
            return False
            
    except Exception as e:
        logger.error(f"真实数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        test_path = "data/processed/real_test_features.h5"
        if Path(test_path).exists():
            Path(test_path).unlink()
            logger.info("已清理真实数据测试文件")

def main():
    """主测试流程"""
    print("="*60)
    print("🧪 HDF5增量处理系统测试")
    print("="*60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1: HDF5基本功能
    print("\n📋 测试1: HDF5基本功能")
    print("-" * 40)
    if test_hdf5_basic_functionality():
        success_count += 1
        print("✅ HDF5基本功能测试通过")
    else:
        print("❌ HDF5基本功能测试失败")
    
    # 测试2: 真实数据处理
    print("\n📋 测试2: 真实数据处理")
    print("-" * 40)
    if test_real_data_processing():
        success_count += 1
        print("✅ 真实数据处理测试通过")
    else:
        print("❌ 真实数据处理测试失败")
    
    # 总结
    print("\n" + "="*60)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    if success_count == total_tests:
        print("🎉 所有测试通过！HDF5增量处理系统工作正常")
        return True
    else:
        print("⚠️  部分测试失败，需要修复问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 