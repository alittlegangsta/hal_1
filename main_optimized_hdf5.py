#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版HDF5增量处理系统 - 2850-2950ft深度范围专用
内存高效的测井数据窜槽检测处理流程
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import gc
from typing import Dict, List, Tuple

# 添加src目录到Python路径
sys.path.append('src')

from src.feature_engineering import IncrementalFeatureEngineer
from src.model import HybridChannelingModel
from src.hdf5_manager import HDF5DataManager
from src.data_loader import DataLoader
from src.signal_processing import SignalProcessor
from src.plot_config import setup_matplotlib

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_hdf5_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_receiver_sector(data_loader: DataLoader, 
                                  signal_processor: SignalProcessor,
                                  filtered_cast: Dict,
                                  filtered_xsilmr: Dict,
                                  receiver_id: int = 7,
                                  sector_idx: int = 0) -> Tuple[int, str]:
    """
    处理单个接收器的单个方位扇区
    
    Args:
        data_loader: 数据加载器
        signal_processor: 信号处理器
        filtered_cast: 筛选后的CAST数据
        filtered_xsilmr: 筛选后的XSILMR数据
        receiver_id: 接收器ID
        sector_idx: 扇区索引
        
    Returns:
        (样本数, HDF5文件路径)
    """
    # 配置参数
    depth_range = (2850.0, 2950.0)
    hdf5_path = f"data/processed/receiver_{receiver_id}_sector_{sector_idx}_features.h5"
    
    logger.info(f"处理接收器 {receiver_id} 扇区 {sector_idx}...")
    
    # 确保输出目录存在
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # 删除旧文件
    if Path(hdf5_path).exists():
        Path(hdf5_path).unlink()
    
    # 获取方位角数据
    side_keys = ['SideA', 'SideB', 'SideC', 'SideD', 
                'SideE', 'SideF', 'SideG', 'SideH']
    
    if receiver_id not in filtered_xsilmr:
        logger.error(f"接收器 {receiver_id} 不存在")
        return 0, hdf5_path
    
    side_key = side_keys[sector_idx]
    if side_key not in filtered_xsilmr[receiver_id]:
        logger.error(f"接收器 {receiver_id} 的 {side_key} 数据不存在")
        return 0, hdf5_path
    
    # 获取数据
    waveforms = filtered_xsilmr[receiver_id][side_key]  # (1024, n_depths)
    depths = filtered_xsilmr[receiver_id]['AbsoluteDepth']
    
    # 创建标签
    cast_depths = filtered_cast['Depth']
    cast_zc = filtered_cast['Zc']
    
    # 计算方位角范围
    sector_size = 360 // 8  # 45度每扇区
    start_angle = sector_idx * sector_size
    end_angle = (sector_idx + 1) * sector_size
    
    # 方位角索引 (每2度一个)
    start_idx = start_angle // 2
    end_idx = end_angle // 2
    
    # 获取扇区数据
    azimuth_zc = cast_zc[start_idx:end_idx, :]
    channeling_mask = azimuth_zc < 2.5
    channeling_ratios = np.mean(channeling_mask, axis=0)
    
    # 插值到XSILMR深度
    sector_labels = np.interp(depths, cast_depths, channeling_ratios)
    
    # 计算实际样本数
    total_samples = min(len(depths), len(sector_labels), waveforms.shape[1])
    
    if total_samples == 0:
        logger.warning(f"接收器 {receiver_id} 扇区 {sector_idx} 没有有效样本")
        return 0, hdf5_path
    
    logger.info(f"预期处理 {total_samples} 个样本")
    
    # 创建HDF5管理器
    manager = HDF5DataManager(hdf5_path, mode='w')
    manager.create_dataset_structure(
        total_samples=total_samples,
        image_shape=(127, 1024),
        vector_dim=8,
        chunk_size=min(50, total_samples)
    )
    
    # 创建批处理器
    from src.hdf5_manager import BatchProcessor
    batch_processor = BatchProcessor(manager, batch_size=min(20, total_samples))
    
    # 处理数据
    processed_count = 0
    for depth_idx in range(total_samples):
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
                sector_idx
            ], dtype=np.float32)
            
            # 标签和元数据
            label = sector_labels[depth_idx]
            metadata = (depths[depth_idx], receiver_id, sector_idx, depth_idx)
            
            # 添加到批处理器
            batch_processor.add_sample(scalogram, vector_features, label, metadata)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                logger.info(f"  已处理 {processed_count}/{total_samples} 个深度点")
                
        except Exception as e:
            logger.warning(f"处理深度点 {depth_idx} 时出错: {e}")
            continue
    
    # 完成处理
    actual_samples = batch_processor.finalize()
    
    # 清理内存
    del batch_processor
    del manager
    gc.collect()
    
    logger.info(f"接收器 {receiver_id} 扇区 {sector_idx} 处理完成: {actual_samples} 样本")
    
    return actual_samples, hdf5_path

def merge_hdf5_files(file_paths: List[str], output_path: str) -> int:
    """
    合并多个HDF5文件到一个文件中
    
    Args:
        file_paths: 输入HDF5文件路径列表
        output_path: 输出HDF5文件路径
        
    Returns:
        合并后的总样本数
    """
    logger.info(f"合并 {len(file_paths)} 个HDF5文件...")
    
    # 计算总样本数
    total_samples = 0
    for file_path in file_paths:
        if Path(file_path).exists():
            manager = HDF5DataManager(file_path, mode='r')
            info = manager.get_data_info()
            total_samples += info['total_samples']
            logger.info(f"  {file_path}: {info['total_samples']} 样本")
    
    if total_samples == 0:
        logger.error("没有有效的样本可以合并")
        return 0
    
    logger.info(f"总样本数: {total_samples}")
    
    # 创建输出文件
    output_manager = HDF5DataManager(output_path, mode='w')
    output_manager.create_dataset_structure(
        total_samples=total_samples,
        image_shape=(127, 1024),
        vector_dim=8,
        chunk_size=min(100, total_samples)
    )
    
    # 合并数据
    current_idx = 0
    for file_path in file_paths:
        if not Path(file_path).exists():
            continue
            
        # 读取源文件
        source_manager = HDF5DataManager(file_path, mode='r')
        source_info = source_manager.get_data_info()
        n_samples = source_info['total_samples']
        
        if n_samples == 0:
            continue
        
        # 分批读取和写入
        batch_size = min(100, n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = source_manager.read_batch(start_idx, end_idx - start_idx)
            
            # 写入到输出文件
            output_manager.write_batch(
                current_idx,
                batch_data['image_features'],
                batch_data['vector_features'],
                batch_data['labels'],
                batch_data.get('metadata')
            )
            
            current_idx += len(batch_data['image_features'])
        
        logger.info(f"  已合并 {file_path}: {n_samples} 样本")
    
    # 清理
    del output_manager
    gc.collect()
    
    logger.info(f"合并完成: 总共 {current_idx} 样本")
    return current_idx

def main():
    """主处理流程"""
    print("="*80)
    print("🚀 优化版HDF5增量处理系统")
    print("="*80)
    print("📋 处理范围: 2850-2950ft (100ft完整深度范围)")
    print("🔧 处理策略: 全接收器(1-13) + 全方位角(A-H) + 最终合并")
    print("⚡ 优化目标: 内存高效 + 数据完整性 + 最大覆盖度")
    print("="*80)
    
    try:
        # 配置matplotlib
        setup_matplotlib()
        
        # 创建输出目录
        output_dirs = ['data/processed', 'outputs/models', 'outputs/figures', 'outputs/logs']
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # 1. 加载和筛选数据
        logger.info("步骤1: 加载和筛选基础数据")
        print("\n📊 步骤1: 数据加载和筛选")
        print("-" * 60)
        
        data_loader = DataLoader()
        signal_processor = SignalProcessor()
        
        # 加载数据
        cast_data = data_loader.load_cast_data()
        xsilmr_data = data_loader.load_xsilmr_data()
        
        # 筛选深度范围
        filtered_cast, filtered_xsilmr = data_loader.filter_depth_range(
            min_depth=2850.0,
            max_depth=2950.0
        )
        
        # 计算绝对深度
        filtered_xsilmr = data_loader.calculate_absolute_depths(filtered_xsilmr)
        
        # 获取一个可用接收器的深度点数作为参考
        reference_receiver = None
        for r_id in range(1, 14):
            if r_id in filtered_xsilmr:
                reference_receiver = r_id
                break
        
        xsilmr_depth_count = len(filtered_xsilmr[reference_receiver]['Depth']) if reference_receiver else 0
        logger.info(f"筛选后数据: CAST={len(filtered_cast['Depth'])}点, XSILMR={xsilmr_depth_count}点 (基于接收器{reference_receiver})")
        
        # 2. 分块处理
        logger.info("步骤2: 分块增量处理")
        print("\n🔄 步骤2: 分块增量处理")
        print("-" * 60)
        
        # 处理配置：处理所有接收器和所有方位
        target_receivers = list(range(1, 14))  # 所有13个接收器 (1-13)
        target_sectors = list(range(8))  # 所有8个方位扇区 (A-H)
        
        hdf5_files = []
        total_processed_samples = 0
        
        for receiver_id in target_receivers:
            for sector_idx in target_sectors:
                print(f"  处理接收器 {receiver_id} 方位扇区 {sector_idx}...")
                
                samples, file_path = process_single_receiver_sector(
                    data_loader, signal_processor,
                    filtered_cast, filtered_xsilmr,
                    receiver_id, sector_idx
                )
                
                if samples > 0:
                    hdf5_files.append(file_path)
                    total_processed_samples += samples
                    print(f"    ✅ 成功处理 {samples} 样本")
                else:
                    print(f"    ⚠️ 跳过（无有效样本）")
                
                # 强制垃圾回收
                gc.collect()
        
        processing_time = time.time() - start_time
        
        # 3. 合并HDF5文件
        logger.info("步骤3: 合并HDF5文件")
        print("\n🔗 步骤3: 合并HDF5文件")
        print("-" * 60)
        
        if len(hdf5_files) > 0:
            final_hdf5_path = "data/processed/final_features_2850_2950.h5"
            
            # 删除旧的合并文件
            if Path(final_hdf5_path).exists():
                Path(final_hdf5_path).unlink()
            
            merged_samples = merge_hdf5_files(hdf5_files, final_hdf5_path)
            
            # 清理临时文件
            for file_path in hdf5_files:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.info(f"清理临时文件: {file_path}")
            
            print(f"✅ 成功合并 {merged_samples} 样本到 {final_hdf5_path}")
            
            # 4. 检查合并后的数据
            logger.info("步骤4: 验证合并数据")
            print("\n🔍 步骤4: 数据验证")
            print("-" * 60)
            
            final_manager = HDF5DataManager(final_hdf5_path, mode='r')
            final_info = final_manager.get_data_info()
            
            print(f"📋 最终数据集信息:")
            print(f"   • 总样本数: {final_info['total_samples']:,}")
            print(f"   • 文件大小: {final_info['file_size_mb']:.1f} MB")
            print(f"   • 图像特征形状: {final_info['image_shape']}")
            print(f"   • 数值特征维度: {final_info['vector_dim']}")
            
            # 读取样本进行统计
            sample_data = final_manager.read_batch(0, min(100, final_info['total_samples']))
            labels = sample_data['labels']
            
            print(f"📊 标签统计 (基于前{len(labels)}个样本):")
            print(f"   • 平均窜槽比例: {np.mean(labels):.3f}")
            print(f"   • 标准差: {np.std(labels):.3f}")
            print(f"   • 最小值: {np.min(labels):.3f}")
            print(f"   • 最大值: {np.max(labels):.3f}")
            print(f"   • 中度窜槽样本 (≥50%): {np.mean(labels >= 0.5):.1%}")
            
            # 5. 模型训练演示（可选）
            print("\n🤖 步骤5: 模型训练演示")
            print("-" * 60)
            
            # 创建模型
            model = HybridChannelingModel(image_shape=(127, 1024), vector_dim=8)
            
            # 训练模型（快速版本）
            print("开始模型训练（演示版：10个epoch）...")
            training_start = time.time()
            
            history = model.train_from_hdf5(
                hdf5_path=final_hdf5_path,
                epochs=10,  # 快速演示
                batch_size=16,
                test_size=0.2,
                val_size=0.1
            )
            
            training_time = time.time() - training_start
            
            # 保存模型
            model.save_model(
                'outputs/models/optimized_channeling_model.h5',
                'outputs/models/optimized_scaler.pkl'
            )
            
            # 绘制训练历史
            model.plot_training_history('outputs/figures/optimized_training_history.png')
            
            print(f"✅ 模型训练完成，耗时: {training_time:.1f}秒")
            
            # 导出最终摘要
            final_manager.export_summary('outputs/logs/final_dataset_summary.txt')
            
        else:
            logger.error("没有成功处理的数据，无法继续")
            return False
        
        # 总结报告
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 优化版HDF5增量处理完成！")
        print("="*80)
        
        print(f"📊 处理统计:")
        print(f"   • 处理深度范围: 2850-2950ft")
        print(f"   • 接收器数量: {len(target_receivers)} 个 (R1-R13)")
        print(f"   • 方位扇区数量: {len(target_sectors)} 个")
        print(f"   • 总处理组合: {len(target_receivers) * len(target_sectors)} 个")
        print(f"   • 总样本数: {merged_samples:,}")
        print(f"   • 处理耗时: {processing_time:.1f}秒")
        print(f"   • 训练耗时: {training_time:.1f}秒")
        print(f"   • 总耗时: {total_time:.1f}秒")
        
        print(f"\n📁 输出文件:")
        print(f"   • 最终数据集: {final_hdf5_path}")
        print(f"   • 训练模型: outputs/models/optimized_channeling_model.h5")
        print(f"   • 数据摘要: outputs/logs/final_dataset_summary.txt")
        print(f"   • 训练图表: outputs/figures/optimized_training_history.png")
        
        print(f"\n🎯 关键优势:")
        print(f"   ✓ 内存高效: 分块处理避免内存溢出")
        print(f"   ✓ 全面覆盖: 处理全部13个接收器的8个方位组合")
        print(f"   ✓ 数据压缩: HDF5格式大幅减少存储空间")
        print(f"   ✓ 可扩展性: 支持任意深度范围和接收器组合")
        print(f"   ✓ 模型训练: 直接从HDF5流式训练深度学习模型")
        
        return True
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理内存
        gc.collect()

if __name__ == "__main__":
    print("优化版测井数据HDF5增量处理系统")
    print("="*50)
    
    success = main()
    
    if success:
        print("\n🎉 处理成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败！")
        sys.exit(1) 