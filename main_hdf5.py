#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测井数据窜槽检测项目 - HDF5增量处理主程序
使用内存高效的HDF5存储方案处理大规模测井数据
深度范围：2850-2950ft (100ft范围以快速获得结果)
"""

import os
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
from src.visualization import DataVisualizer
from src.plot_config import setup_matplotlib

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hdf5_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主处理流程"""
    print("="*80)
    print("🚀 测井数据窜槽检测项目 - HDF5增量处理")
    print("="*80)
    print("📋 处理范围: 2850-2950ft (100ft精确分析)")
    print("💾 技术方案: HDF5增量存储 + 内存高效训练")
    print("⚡ 优化目标: 快速结果 + 内存友好")
    print("="*80)
    
    # 配置matplotlib
    setup_matplotlib()
    
    # 创建输出目录
    output_dirs = ['data/processed', 'outputs/models', 'outputs/figures', 'outputs/logs']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 配置参数
    config = {
        'depth_range': (2850.0, 2950.0),  # 100ft精确范围
        'azimuth_sectors': 8,
        'batch_size': 50,
        'hdf5_path': 'data/processed/features_2850_2950.h5',
        'model_epochs': 50,  # 减少epoch数以快速获得结果
        'train_batch_size': 32
    }
    
    logger.info(f"处理配置: {config}")
    
    try:
        # 步骤1: 增量特征工程
        print("\n" + "="*60)
        print("📊 步骤1: 增量特征工程处理")
        print("="*60)
        
        start_time = time.time()
        
        feature_engineer = IncrementalFeatureEngineer(
            depth_range=config['depth_range'],
            azimuth_sectors=config['azimuth_sectors'],
            batch_size=config['batch_size']
        )
        
        total_samples = feature_engineer.generate_features_to_hdf5(config['hdf5_path'])
        
        feature_time = time.time() - start_time
        logger.info(f"特征工程完成: {total_samples} 样本, 耗时 {feature_time:.1f}秒")
        
        # 检查HDF5文件信息
        print_hdf5_info(config['hdf5_path'])
        
        # 步骤2: 模型训练
        print("\n" + "="*60)
        print("🤖 步骤2: 深度学习模型训练")
        print("="*60)
        
        start_time = time.time()
        
        model = HybridChannelingModel(
            image_shape=(127, 1024),
            vector_dim=8
        )
        
        # 从HDF5文件训练模型
        history = model.train_from_hdf5(
            hdf5_path=config['hdf5_path'],
            epochs=config['model_epochs'],
            batch_size=config['train_batch_size'],
            test_size=0.2,
            val_size=0.1
        )
        
        train_time = time.time() - start_time
        logger.info(f"模型训练完成: 耗时 {train_time:.1f}秒")
        
        # 步骤3: 结果可视化和分析
        print("\n" + "="*60)
        print("📈 步骤3: 结果分析和可视化")
        print("="*60)
        
        # 绘制训练历史
        model.plot_training_history('outputs/figures/hdf5_training_history.png')
        
        # 数据集统计分析
        analyze_dataset_statistics(config['hdf5_path'])
        
        # 模型性能分析
        evaluate_model_performance(model, config['hdf5_path'])
        
        # 步骤4: 保存模型
        print("\n" + "="*60)
        print("💾 步骤4: 保存模型和结果")
        print("="*60)
        
        model.save_model(
            'outputs/models/hdf5_channeling_model.h5',
            'outputs/models/hdf5_scaler.pkl'
        )
        
        # 导出HDF5数据集摘要
        hdf5_manager = HDF5DataManager(config['hdf5_path'], mode='r')
        hdf5_manager.export_summary('outputs/logs/hdf5_dataset_summary.txt')
        
        # 总结报告
        print("\n" + "="*80)
        print("✅ HDF5增量处理完成！")
        print("="*80)
        
        total_time = feature_time + train_time
        print(f"📊 处理统计:")
        print(f"   • 总样本数: {total_samples:,}")
        print(f"   • 深度范围: {config['depth_range'][0]}-{config['depth_range'][1]}ft")
        print(f"   • 特征工程耗时: {feature_time:.1f}秒")
        print(f"   • 模型训练耗时: {train_time:.1f}秒")
        print(f"   • 总耗时: {total_time:.1f}秒")
        
        hdf5_info = HDF5DataManager(config['hdf5_path'], mode='r').get_data_info()
        print(f"   • HDF5文件大小: {hdf5_info['file_size_mb']:.1f}MB")
        
        print(f"\n📁 输出文件:")
        print(f"   • HDF5数据集: {config['hdf5_path']}")
        print(f"   • 训练模型: outputs/models/hdf5_channeling_model.h5")
        print(f"   • 标准化器: outputs/models/hdf5_scaler.pkl")
        print(f"   • 训练图表: outputs/figures/hdf5_training_history.png")
        print(f"   • 数据摘要: outputs/logs/hdf5_dataset_summary.txt")
        
        print(f"\n🎯 主要优势:")
        print(f"   ✓ 内存高效: 增量处理，避免内存溢出")
        print(f"   ✓ 可扩展性: 支持任意大小的数据集")
        print(f"   ✓ 数据压缩: HDF5格式节省存储空间")
        print(f"   ✓ 快速训练: 直接从磁盘流式读取")
        print(f"   ✓ 可复现性: 完整的数据处理流程")
        
        return True
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理内存
        gc.collect()

def print_hdf5_info(hdf5_path: str):
    """打印HDF5文件信息"""
    print(f"\n📋 HDF5数据集信息:")
    print("-" * 40)
    
    hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
    info = hdf5_manager.get_data_info()
    
    print(f"文件路径: {hdf5_path}")
    print(f"文件大小: {info['file_size_mb']:.2f} MB")
    print(f"总样本数: {info['total_samples']:,}")
    print(f"图像特征形状: {info['image_shape']}")
    print(f"数值特征维度: {info['vector_dim']}")
    print(f"数据集列表: {', '.join(info['datasets'])}")
    
    # 计算压缩率
    uncompressed_size = (
        info['total_samples'] * (
            np.prod(info['image_shape']) * 4 +  # 图像特征 (float32)
            info['vector_dim'] * 4 +            # 数值特征 (float32)
            4                                   # 标签 (float32)
        )
    ) / (1024 * 1024)  # 转换为MB
    
    compression_ratio = uncompressed_size / info['file_size_mb']
    print(f"压缩率: {compression_ratio:.1f}x ({uncompressed_size:.1f}MB -> {info['file_size_mb']:.1f}MB)")

def analyze_dataset_statistics(hdf5_path: str):
    """分析数据集统计信息"""
    print(f"\n📊 数据集统计分析:")
    print("-" * 40)
    
    hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
    
    # 读取少量数据进行统计分析
    sample_size = min(1000, hdf5_manager.get_data_info()['total_samples'])
    sample_data = hdf5_manager.read_batch(0, sample_size)
    
    labels = sample_data['labels']
    vector_features = sample_data['vector_features']
    
    print(f"标签统计 (基于{sample_size}个样本):")
    print(f"  平均窜槽比例: {np.mean(labels):.3f}")
    print(f"  标准差: {np.std(labels):.3f}")
    print(f"  最小值: {np.min(labels):.3f}")
    print(f"  最大值: {np.max(labels):.3f}")
    print(f"  中位数: {np.median(labels):.3f}")
    
    # 窜槽严重程度分布
    mild_ratio = np.mean(labels >= 0.3)
    moderate_ratio = np.mean(labels >= 0.5)
    severe_ratio = np.mean(labels >= 0.7)
    
    print(f"窜槽严重程度分布:")
    print(f"  轻度窜槽 (≥30%): {mild_ratio:.1%}")
    print(f"  中度窜槽 (≥50%): {moderate_ratio:.1%}")
    print(f"  重度窜槽 (≥70%): {severe_ratio:.1%}")
    
    print(f"数值特征统计:")
    feature_names = ['max_amplitude', 'rms_amplitude', 'energy', 'zero_crossings',
                    'dominant_frequency', 'spectral_centroid', 'receiver_id', 'sector_id']
    
    for i, name in enumerate(feature_names):
        if i < vector_features.shape[1]:
            values = vector_features[:, i]
            print(f"  {name}: 均值={np.mean(values):.2e}, 标准差={np.std(values):.2e}")

def evaluate_model_performance(model: HybridChannelingModel, hdf5_path: str):
    """评估模型性能"""
    print(f"\n🎯 模型性能评估:")
    print("-" * 40)
    
    # 创建测试数据生成器
    from src.model import HDF5DataGenerator
    
    hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
    total_samples = hdf5_manager.get_data_info()['total_samples']
    
    # 使用最后20%的数据作为测试集
    test_size = int(total_samples * 0.2)
    test_indices = list(range(total_samples - test_size, total_samples))
    
    test_generator = HDF5DataGenerator(hdf5_path, test_indices, batch_size=32, shuffle=False)
    
    # 评估模型
    test_metrics = model.model.evaluate(test_generator, verbose=0)
    
    print("测试集评估结果:")
    for name, value in zip(model.model.metrics_names, test_metrics):
        print(f"  {name}: {value:.4f}")
    
    # 获取预测结果进行详细分析
    print("正在生成预测结果用于详细分析...")
    
    # 读取一小批测试数据进行预测分析
    test_batch = hdf5_manager.read_batch(total_samples - 100, 100)
    
    # 预处理图像数据
    image_features = test_batch['image_features']
    image_features = image_features[..., np.newaxis]
    for i in range(len(image_features)):
        max_val = np.max(image_features[i])
        if max_val > 1e-8:
            image_features[i] = image_features[i] / max_val
    
    # 预处理数值特征
    vector_features = model.scaler.transform(test_batch['vector_features'])
    
    # 预测
    y_pred = model.model.predict([image_features, vector_features], verbose=0).flatten()
    y_true = test_batch['labels']
    
    # 绘制预测结果
    model.plot_predictions(y_true, y_pred, 'outputs/figures/hdf5_predictions.png')
    
    # 计算详细指标
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"详细性能指标 (基于{len(y_true)}个测试样本):")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 预测精度分析
    abs_errors = np.abs(y_pred - y_true)
    print(f"预测精度分析:")
    print(f"  平均绝对误差: {np.mean(abs_errors):.3f}")
    print(f"  误差标准差: {np.std(abs_errors):.3f}")
    print(f"  90%样本误差 < {np.percentile(abs_errors, 90):.3f}")
    print(f"  95%样本误差 < {np.percentile(abs_errors, 95):.3f}")

def demo_hdf5_capabilities():
    """演示HDF5增量处理的核心功能"""
    print("\n" + "="*60)
    print("🔧 HDF5增量处理技术演示")
    print("="*60)
    
    demo_path = "data/processed/demo_features.h5"
    
    # 创建演示用的小规模HDF5文件
    print("创建演示HDF5数据集...")
    demo_manager = HDF5DataManager(demo_path, mode='w')
    demo_manager.create_dataset_structure(
        total_samples=1000,
        image_shape=(127, 1024),
        vector_dim=8,
        chunk_size=100
    )
    
    # 演示批量写入
    print("演示批量数据写入...")
    for i in range(0, 1000, 100):
        # 生成模拟数据
        batch_images = np.random.random((100, 127, 1024)).astype(np.float32)
        batch_vectors = np.random.random((100, 8)).astype(np.float32)
        batch_labels = np.random.random(100).astype(np.float32)
        
        demo_manager.write_batch(i, batch_images, batch_vectors, batch_labels)
        print(f"  已写入批次 {i//100 + 1}/10")
    
    # 演示数据读取
    print("演示数据读取...")
    demo_manager.mode = 'r'
    
    # 读取部分数据
    batch_data = demo_manager.read_batch(0, 50)
    print(f"  读取50个样本: 图像形状={batch_data['image_features'].shape}")
    
    # 创建数据迭代器
    print("演示数据迭代器...")
    iterator = demo_manager.create_data_iterator(batch_size=32, shuffle=True)
    for i, batch in enumerate(iterator):
        if i >= 3:  # 只演示前3个批次
            break
        print(f"  批次{i+1}: 图像={batch['image_features'].shape}, 标签={batch['labels'].shape}")
    
    # 清理演示文件
    Path(demo_path).unlink(missing_ok=True)
    print("演示完成，已清理临时文件")

if __name__ == "__main__":
    print("测井数据HDF5增量处理系统")
    print("="*50)
    
    # 可选: 运行技术演示
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_hdf5_capabilities()
    
    # 运行主处理流程
    success = main()
    
    if success:
        print("\n🎉 处理成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败！")
        sys.exit(1) 