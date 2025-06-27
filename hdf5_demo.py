"""
HDF5增量存储演示脚本
演示如何使用增量特征工程器进行内存高效的大规模数据处理
"""
import os
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import platform
import time

# 配置中文字体
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    elif system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']
    
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            test_fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(test_fig)
            print(f"成功设置中文字体: {font}")
            break
        except:
            continue
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        print("使用默认字体配置")
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体设置
setup_chinese_font()

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

# 导入项目模块
from src.feature_engineering import IncrementalFeatureEngineer
from src.hdf5_manager import HDF5DataLoader, HDF5DataManager
from src.model import HybridChannelingModel

def main():
    """主函数 - 演示HDF5增量存储"""
    logger.info("=" * 60)
    logger.info("HDF5增量存储演示开始")
    logger.info("深度范围: 2850-2950 ft (100 ft)")
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # HDF5文件路径
    hdf5_path = "data/processed/features_2850_2950.h5"
    
    try:
        # ========== 阶段一：增量特征工程 ==========
        logger.info("阶段一：增量特征工程 (HDF5存储)")
        
        # 初始化增量特征工程器
        incremental_engineer = IncrementalFeatureEngineer(
            depth_range=(2850.0, 2950.0),  # 100ft范围
            azimuth_sectors=8,
            batch_size=50  # 批处理大小
        )
        
        # 估算数据量
        estimated_samples = incremental_engineer.estimate_total_samples()
        logger.info(f"预估样本数: {estimated_samples}")
        
        # 生成特征并存储到HDF5
        start_time = time.time()
        actual_samples = incremental_engineer.generate_features_to_hdf5(hdf5_path)
        processing_time = time.time() - start_time
        
        logger.info(f"增量特征工程完成:")
        logger.info(f"  实际样本数: {actual_samples}")
        logger.info(f"  处理时间: {processing_time:.2f} 秒")
        logger.info(f"  平均速度: {actual_samples/processing_time:.2f} 样本/秒")
        
        # 检查HDF5文件
        hdf5_file = Path(hdf5_path)
        if hdf5_file.exists():
            file_size_gb = hdf5_file.stat().st_size / (1024**3)
            logger.info(f"HDF5文件大小: {file_size_gb:.2f} GB")
        
        # ========== 阶段二：数据验证 ==========
        logger.info("阶段二：验证HDF5数据")
        
        # 使用HDF5数据加载器验证数据
        with HDF5DataLoader(hdf5_path) as data_loader:
            # 获取数据集信息
            manager = data_loader.manager
            info = manager.get_dataset_info()
            total_samples = manager.get_total_samples()
            
            logger.info(f"数据集验证:")
            logger.info(f"  总样本数: {total_samples}")
            logger.info(f"  图像形状: {info.get('image_shape', 'N/A')}")
            logger.info(f"  向量维度: {info.get('vector_dim', 'N/A')}")
            
            # 导出数据摘要
            manager.export_summary()
            
            # 示例：读取前几个样本
            logger.info("读取示例样本:")
            for i in range(min(3, total_samples)):
                image, vector, label, metadata = data_loader.get_sample(i)
                depth, receiver_id, azimuth_sector, sample_id = metadata
                logger.info(f"  样本 {i}: 深度={depth:.1f}ft, "
                           f"接收器={int(receiver_id)}, 方位={int(azimuth_sector)}, "
                           f"标签={label:.4f}")
        
        # ========== 阶段三：批量数据读取演示 ==========
        logger.info("阶段三：批量数据读取演示")
        
        with HDF5DataLoader(hdf5_path) as data_loader:
            batch_count = 0
            sample_count = 0
            
            # 使用批量生成器
            for images, vectors, labels in data_loader.get_batch_generator(
                batch_size=32, shuffle=False
            ):
                batch_count += 1
                sample_count += len(images)
                
                if batch_count <= 3:  # 只显示前3个批次的信息
                    logger.info(f"  批次 {batch_count}: "
                               f"图像形状={images.shape}, "
                               f"向量形状={vectors.shape}, "
                               f"标签形状={labels.shape}")
                    logger.info(f"    标签范围: {np.min(labels):.4f} - {np.max(labels):.4f}")
                
                if batch_count >= 5:  # 只演示前5个批次
                    break
            
            logger.info(f"批量读取完成: 处理了 {batch_count} 个批次, 共 {sample_count} 个样本")
        
        # ========== 阶段四：可视化分析 ==========
        logger.info("阶段四：生成可视化分析")
        
        # 创建分析图表
        create_analysis_plots(hdf5_path, figures_dir)
        
        # ========== 阶段五：模型训练演示 (可选) ==========
        if input("是否进行模型训练演示? (y/n): ").lower() == 'y':
            logger.info("阶段五：模型训练演示")
            
            with HDF5DataLoader(hdf5_path) as data_loader:
                # 加载小批量数据进行训练演示
                images, vectors, labels = [], [], []
                sample_count = 0
                
                for batch_images, batch_vectors, batch_labels in data_loader.get_batch_generator(
                    batch_size=32, shuffle=True
                ):
                    images.extend(batch_images)
                    vectors.extend(batch_vectors)
                    labels.extend(batch_labels)
                    sample_count += len(batch_images)
                    
                    if sample_count >= 500:  # 限制训练样本数以节省时间
                        break
                
                images = np.array(images)
                vectors = np.array(vectors)
                labels = np.array(labels)
                
                logger.info(f"训练数据: {len(images)} 个样本")
                
                # 创建模型并训练
                model = HybridChannelingModel(
                    image_shape=images[0].shape, 
                    vector_dim=len(vectors[0])
                )
                
                train_data, test_data = model.prepare_data(images, vectors, labels)
                
                # 快速训练演示
                history = model.train(
                    train_data=train_data,
                    val_data=test_data,
                    epochs=10,  # 少量训练轮数
                    batch_size=32
                )
                
                logger.info("模型训练演示完成")
        
        logger.info("=" * 60)
        logger.info("HDF5增量存储演示完成")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

def create_analysis_plots(hdf5_path: str, figures_dir: Path):
    """创建数据分析图表"""
    logger.info("生成数据分析图表...")
    
    with HDF5DataLoader(hdf5_path) as data_loader:
        # 加载所有标签用于分析
        manager = data_loader.manager
        labels = manager.datasets['labels'][:]
        metadata = manager.datasets['metadata'][:]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('测井数据窜槽分析 (2850-2950 ft)', fontsize=16)
        
        # 1. 标签分布直方图
        axes[0, 0].hist(labels, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('窜槽比例分布')
        axes[0, 0].set_xlabel('窜槽比例')
        axes[0, 0].set_ylabel('样本数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 接收器分布
        receiver_ids = metadata[:, 1].astype(int)
        unique_receivers, counts = np.unique(receiver_ids, return_counts=True)
        axes[0, 1].bar(unique_receivers, counts, color='green', alpha=0.7)
        axes[0, 1].set_title('接收器样本分布')
        axes[0, 1].set_xlabel('接收器ID')
        axes[0, 1].set_ylabel('样本数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 方位角分布
        azimuth_sectors = metadata[:, 2].astype(int)
        unique_sectors, counts = np.unique(azimuth_sectors, return_counts=True)
        axes[1, 0].bar(unique_sectors, counts, color='orange', alpha=0.7)
        axes[1, 0].set_title('方位角扇区分布')
        axes[1, 0].set_xlabel('方位角扇区')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 深度vs窜槽比例散点图
        depths = metadata[:, 0]
        # 为了可视化，只选择部分样本
        sample_indices = np.random.choice(len(depths), size=min(1000, len(depths)), replace=False)
        axes[1, 1].scatter(depths[sample_indices], labels[sample_indices], 
                          alpha=0.5, s=2, color='red')
        axes[1, 1].set_title('深度 vs 窜槽比例')
        axes[1, 1].set_xlabel('深度 (ft)')
        axes[1, 1].set_ylabel('窜槽比例')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = figures_dir / "channeling_analysis_2850_2950.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"分析图表已保存: {save_path}")
        
        # 生成统计报告
        report_path = figures_dir.parent / "analysis_report_2850_2950.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("测井数据窜槽分析报告 (2850-2950 ft)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总样本数: {len(labels)}\n")
            f.write(f"深度范围: {np.min(depths):.1f} - {np.max(depths):.1f} ft\n")
            f.write(f"窜槽比例范围: {np.min(labels):.4f} - {np.max(labels):.4f}\n")
            f.write(f"窜槽比例均值: {np.mean(labels):.4f}\n")
            f.write(f"窜槽比例标准差: {np.std(labels):.4f}\n\n")
            
            # 窜槽严重程度分类
            low_channeling = np.sum(labels <= 0.1)
            medium_channeling = np.sum((labels > 0.1) & (labels <= 0.3))
            high_channeling = np.sum(labels > 0.3)
            
            f.write("窜槽严重程度分类:\n")
            f.write(f"  低窜槽 (≤0.1): {low_channeling} ({low_channeling/len(labels)*100:.1f}%)\n")
            f.write(f"  中窜槽 (0.1-0.3): {medium_channeling} ({medium_channeling/len(labels)*100:.1f}%)\n")
            f.write(f"  高窜槽 (>0.3): {high_channeling} ({high_channeling/len(labels)*100:.1f}%)\n")
        
        logger.info(f"分析报告已保存: {report_path}")

if __name__ == "__main__":
    main() 