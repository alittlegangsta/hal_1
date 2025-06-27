#!/usr/bin/env python3
"""
窜槽检测完整流水线
一键运行从数据处理到模型训练到结果输出的完整流程
"""
import os
import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import platform

# 设置中文字体
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
            plt.rcParams['axes.unicode_minus'] = False
            # 测试字体
            test_fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(test_fig)
            print(f"成功设置中文字体: {font}")
            return True
        except:
            continue
    
    print("警告: 无法设置中文字体，将使用默认字体")
    return False

def setup_logging(output_dir: Path) -> str:
    """设置日志记录"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return str(log_file)

class CompletePipeline:
    """完整的窜槽检测流水线"""
    
    def __init__(self, 
                 depth_range=(2850.0, 2950.0),
                 data_scale="small",
                 output_dir="outputs"):
        """
        初始化流水线
        
        Args:
            depth_range: 深度范围 (ft)
            data_scale: 数据规模 ("small", "medium", "large")
            output_dir: 输出目录
        """
        self.depth_range = depth_range
        self.data_scale = data_scale
        self.output_dir = Path(output_dir)
        
        # 根据数据规模设置参数
        self.scale_config = {
            "small": {
                "receivers": [1],
                "azimuth_sectors": 2,
                "max_depths_per_sector": 50,
                "batch_size": 20,
                "epochs": 30,
                "description": "快速测试 (1个接收器, 2个扇区, 50个深度点/扇区)"
            },
            "medium": {
                "receivers": [1, 2, 3, 4, 5],
                "azimuth_sectors": 4,
                "max_depths_per_sector": 100,
                "batch_size": 50,
                "epochs": 50,
                "description": "中等规模 (5个接收器, 4个扇区, 100个深度点/扇区)"
            },
            "large": {
                "receivers": list(range(1, 14)),  # 1-13
                "azimuth_sectors": 8,
                "max_depths_per_sector": None,  # 全部深度点
                "batch_size": 100,
                "epochs": 100,
                "description": "完整规模 (13个接收器, 8个扇区, 全部深度点)"
            }
        }
        
        self.config = self.scale_config[data_scale]
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # 设置日志
        self.log_file = setup_logging(self.output_dir)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("=" * 80)
        self.logger.info("窜槽检测完整流水线初始化")
        self.logger.info("=" * 80)
        self.logger.info(f"深度范围: {depth_range[0]}-{depth_range[1]} ft")
        self.logger.info(f"数据规模: {data_scale} - {self.config['description']}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("=" * 80)
    
    def run_complete_pipeline(self):
        """运行完整流水线"""
        start_time = time.time()
        
        try:
            # 阶段1: 数据处理
            hdf5_path = self.stage1_data_processing()
            
            # 阶段2: 数据分析
            self.stage2_data_analysis(hdf5_path)
            
            # 阶段3: 模型训练
            model, history = self.stage3_model_training(hdf5_path)
            
            # 阶段4: 结果输出
            self.stage4_result_output(model, history, hdf5_path)
            
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("🎉 完整流水线执行成功!")
            self.logger.info(f"总耗时: {total_time/60:.2f} 分钟")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"流水线执行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def stage1_data_processing(self) -> str:
        """阶段1: 数据处理 - HDF5增量存储"""
        self.logger.info("\n" + "="*60)
        self.logger.info("阶段1: 数据处理 - HDF5增量存储")
        self.logger.info("="*60)
        
        from src.data_loader import DataLoader
        from src.signal_processing import SignalProcessor
        from src.hdf5_manager import HDF5DataManager, BatchProcessor
        import gc
        
        # HDF5文件路径
        hdf5_path = self.output_dir / "data" / f"features_{self.data_scale}_{int(self.depth_range[0])}_{int(self.depth_range[1])}.h5"
        
        # 1. 加载原始数据
        self.logger.info("加载原始数据...")
        data_loader = DataLoader()
        cast_data = data_loader.load_cast_data()
        xsilmr_data = data_loader.load_xsilmr_data()
        
        # 2. 筛选深度范围
        self.logger.info(f"筛选深度范围: {self.depth_range[0]}-{self.depth_range[1]} ft")
        filtered_cast, filtered_xsilmr = data_loader.filter_depth_range(
            min_depth=self.depth_range[0], max_depth=self.depth_range[1]
        )
        filtered_xsilmr = data_loader.calculate_absolute_depths(filtered_xsilmr)
        
        # 3. 估算样本数
        total_samples = self._estimate_samples(filtered_xsilmr)
        self.logger.info(f"预估样本数: {total_samples}")
        
        # 4. 创建HDF5文件
        hdf5_manager = HDF5DataManager(str(hdf5_path), mode='w')
        hdf5_manager.create_dataset_structure(
            total_samples=total_samples,
            image_shape=(127, 1024),
            vector_dim=8,
            chunk_size=self.config['batch_size']
        )
        
        batch_processor = BatchProcessor(hdf5_manager, batch_size=self.config['batch_size'])
        signal_processor = SignalProcessor()
        
        # 5. 处理数据
        actual_samples = self._process_receivers(
            filtered_cast, filtered_xsilmr, batch_processor, signal_processor
        )
        
        # 6. 完成处理
        final_count = batch_processor.finalize()
        hdf5_manager.close()
        
        self.logger.info(f"数据处理完成: 预期{total_samples}个，实际生成{final_count}个样本")
        self.logger.info(f"HDF5文件: {hdf5_path}")
        self.logger.info(f"文件大小: {hdf5_path.stat().st_size / (1024**2):.2f} MB")
        
        return str(hdf5_path)
    
    def _estimate_samples(self, filtered_xsilmr: dict) -> int:
        """估算样本数"""
        total_samples = 0
        
        for receiver_id in self.config['receivers']:
            if receiver_id not in filtered_xsilmr:
                continue
            
            receiver_data = filtered_xsilmr[receiver_id]
            n_depths = len(receiver_data['AbsoluteDepth'])
            
            if self.config['max_depths_per_sector']:
                n_depths = min(n_depths, self.config['max_depths_per_sector'])
            
            samples_per_receiver = self.config['azimuth_sectors'] * n_depths
            total_samples += samples_per_receiver
        
        return total_samples
    
    def _process_receivers(self, cast_data, xsilmr_data, batch_processor, signal_processor):
        """处理所有接收器的数据"""
        sample_count = 0
        
        for receiver_id in self.config['receivers']:
            if receiver_id not in xsilmr_data:
                self.logger.warning(f"接收器{receiver_id}不存在，跳过")
                continue
            
            self.logger.info(f"处理接收器 {receiver_id}")
            receiver_data = xsilmr_data[receiver_id]
            depths = receiver_data['AbsoluteDepth']
            
            for sector_idx in range(self.config['azimuth_sectors']):
                self.logger.info(f"  方位角扇区 {sector_idx + 1}/{self.config['azimuth_sectors']}")
                
                # 获取CAST数据
                cast_ratios = self._get_sector_cast_data(cast_data, sector_idx)
                
                # 获取XSILMR数据
                side_key = self._get_side_key(sector_idx)
                if side_key not in receiver_data:
                    self.logger.warning(f"接收器{receiver_id}缺少{side_key}数据")
                    continue
                
                xsilmr_waves = receiver_data[side_key]
                
                # 处理深度点
                n_depths = xsilmr_waves.shape[1]
                if self.config['max_depths_per_sector']:
                    n_depths = min(n_depths, self.config['max_depths_per_sector'])
                
                sample_count += self._process_depth_points(
                    xsilmr_waves, depths, cast_ratios, receiver_id, sector_idx,
                    sample_count, batch_processor, signal_processor, n_depths
                )
                
                # 强制垃圾回收
                import gc
                gc.collect()
        
        return sample_count
    
    def _get_sector_cast_data(self, cast_data, sector_idx):
        """获取扇区内的CAST数据"""
        n_sectors = self.config['azimuth_sectors']
        angle_per_sector = 360 // n_sectors
        
        start_angle = sector_idx * angle_per_sector
        end_angle = (sector_idx + 1) * angle_per_sector
        
        start_idx = start_angle // 2
        end_idx = end_angle // 2
        
        cast_zc = cast_data['Zc'][start_idx:end_idx, :]
        channeling_mask = cast_zc < 2.5
        channeling_ratios = np.mean(channeling_mask, axis=0)
        
        return channeling_ratios
    
    def _get_side_key(self, sector_idx):
        """根据扇区索引获取数据键"""
        # 简化映射：偶数扇区用SideA，奇数扇区用SideB
        return 'SideA' if sector_idx % 2 == 0 else 'SideB'
    
    def _process_depth_points(self, xsilmr_waves, depths, cast_ratios, receiver_id, 
                            sector_idx, base_sample_id, batch_processor, signal_processor, n_depths):
        """处理深度点"""
        processed_count = 0
        
        for depth_idx in range(n_depths):
            if depth_idx % 50 == 0 and depth_idx > 0:
                self.logger.info(f"    处理深度点 {depth_idx}/{n_depths}")
            
            try:
                # 提取波形
                waveform = xsilmr_waves[:, depth_idx]
                
                # 信号处理
                filtered_waveform = signal_processor.apply_highpass_filter(waveform)
                scalogram = signal_processor.generate_scalogram(filtered_waveform)
                physical_features = signal_processor.extract_physical_features(filtered_waveform)
                
                # 特征向量
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
                
                # 标签
                label = cast_ratios[depth_idx] if depth_idx < len(cast_ratios) else 0.0
                
                # 元数据
                metadata = (
                    depths[depth_idx],
                    receiver_id,
                    sector_idx,
                    base_sample_id + processed_count
                )
                
                # 添加到批处理器
                batch_processor.add_sample(
                    image_feature=scalogram,
                    vector_feature=vector_features,
                    label=label,
                    metadata=metadata
                )
                
                processed_count += 1
                
            except Exception as e:
                self.logger.warning(f"处理深度点{depth_idx}时出错: {e}")
                continue
        
        return processed_count
    
    def stage2_data_analysis(self, hdf5_path: str):
        """阶段2: 数据分析"""
        self.logger.info("\n" + "="*60)
        self.logger.info("阶段2: 数据分析")
        self.logger.info("="*60)
        
        from src.hdf5_manager import HDF5DataLoader
        from src.visualization import DataVisualizer
        
        # 加载数据
        with HDF5DataLoader(hdf5_path) as loader:
            # 导出摘要
            loader.manager.export_summary()
            
            # 获取数据统计
            info = loader.manager.get_dataset_info()
            total_samples = loader.manager.get_total_samples()
            
            self.logger.info(f"数据集信息:")
            self.logger.info(f"  总样本数: {total_samples}")
            self.logger.info(f"  图像形状: {info.get('image_shape', 'N/A')}")
            self.logger.info(f"  向量维度: {info.get('vector_dim', 'N/A')}")
            
            # 数据可视化
            visualizer = DataVisualizer()
            
            # 读取部分数据进行分析
            sample_size = min(1000, total_samples)
            images, vectors, labels = loader.manager.datasets['images'][:sample_size], \
                                    loader.manager.datasets['vectors'][:sample_size], \
                                    loader.manager.datasets['labels'][:sample_size]
            
            # 生成分析图表
            fig_path = self.output_dir / "figures" / f"data_analysis_{self.data_scale}.png"
            visualizer.plot_data_analysis(
                images=images,
                vectors=vectors, 
                labels=labels,
                save_path=str(fig_path)
            )
            
            self.logger.info(f"数据分析图表已保存: {fig_path}")
            self.logger.info(f"数据统计:")
            self.logger.info(f"  窜槽比例均值: {np.mean(labels):.4f}")
            self.logger.info(f"  窜槽比例标准差: {np.std(labels):.4f}")
            self.logger.info(f"  高窜槽样本 (>0.3): {np.sum(labels > 0.3)} ({np.sum(labels > 0.3)/len(labels)*100:.1f}%)")
    
    def stage3_model_training(self, hdf5_path: str):
        """阶段3: 模型训练"""
        self.logger.info("\n" + "="*60)
        self.logger.info("阶段3: 模型训练")
        self.logger.info("="*60)
        
        from src.model import HybridChannelingModel
        
        # 初始化模型
        model = HybridChannelingModel()
        
        # 从HDF5训练
        self.logger.info("开始模型训练...")
        history = model.train_from_hdf5(
            hdf5_path=hdf5_path,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            test_size=0.2,
            val_size=0.1
        )
        
        # 保存模型
        model_path = self.output_dir / "models" / f"channeling_model_{self.data_scale}.h5"
        scaler_path = self.output_dir / "models" / f"scaler_{self.data_scale}.pkl"
        
        model.save_model(str(model_path), str(scaler_path))
        self.logger.info(f"模型已保存: {model_path}")
        self.logger.info(f"标准化器已保存: {scaler_path}")
        
        return model, history
    
    def stage4_result_output(self, model, history, hdf5_path: str):
        """阶段4: 结果输出"""
        self.logger.info("\n" + "="*60)
        self.logger.info("阶段4: 结果输出")
        self.logger.info("="*60)
        
        from src.visualization import DataVisualizer
        from src.hdf5_manager import HDF5DataLoader
        
        visualizer = DataVisualizer()
        
        # 1. 训练历史图
        history_fig = self.output_dir / "figures" / f"training_history_{self.data_scale}.png"
        model.plot_training_history(str(history_fig))
        self.logger.info(f"训练历史图已保存: {history_fig}")
        
        # 2. 模型预测结果
        with HDF5DataLoader(hdf5_path) as loader:
            # 读取测试数据
            total_samples = loader.manager.get_total_samples()
            test_size = min(200, total_samples // 5)  # 取20%作为测试，最多200个样本
            
            images = loader.manager.datasets['images'][-test_size:]
            vectors = loader.manager.datasets['vectors'][-test_size:]
            labels = loader.manager.datasets['labels'][-test_size:]
            
            # 预处理
            images_processed = images[..., np.newaxis]  # 添加通道维度
            
            # 逐样本归一化
            for i in range(len(images_processed)):
                max_val = np.max(images_processed[i])
                if max_val > 1e-8:
                    images_processed[i] = images_processed[i] / max_val
            
            # 标准化向量特征
            vectors_processed = model.scaler.transform(vectors)
            
            # 预测
            predictions = model.model.predict([images_processed, vectors_processed], verbose=0)
            predictions = predictions.flatten()
            
            # 绘制预测结果
            pred_fig = self.output_dir / "figures" / f"predictions_{self.data_scale}.png"
            model.plot_predictions(labels, predictions, str(pred_fig))
            self.logger.info(f"预测结果图已保存: {pred_fig}")
            
            # 计算评估指标
            mse = np.mean((labels - predictions) ** 2)
            mae = np.mean(np.abs(labels - predictions))
            rmse = np.sqrt(mse)
            
            self.logger.info("模型评估结果:")
            self.logger.info(f"  均方误差 (MSE): {mse:.6f}")
            self.logger.info(f"  平均绝对误差 (MAE): {mae:.6f}")
            self.logger.info(f"  均方根误差 (RMSE): {rmse:.6f}")
        
        # 3. 生成完整报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终报告"""
        report_path = self.output_dir / f"pipeline_report_{self.data_scale}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("窜槽检测完整流水线执行报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据规模: {self.data_scale} - {self.config['description']}\n")
            f.write(f"深度范围: {self.depth_range[0]}-{self.depth_range[1]} ft\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"日志文件: {self.log_file}\n\n")
            
            f.write("生成的文件:\n")
            f.write("1. 数据文件:\n")
            for file in (self.output_dir / "data").glob("*.h5"):
                f.write(f"   - {file.name}: {file.stat().st_size / (1024**2):.2f} MB\n")
            
            f.write("\n2. 模型文件:\n")
            for file in (self.output_dir / "models").glob("*"):
                f.write(f"   - {file.name}\n")
            
            f.write("\n3. 图表文件:\n")
            for file in (self.output_dir / "figures").glob("*.png"):
                f.write(f"   - {file.name}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("流水线执行完成!\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"最终报告已生成: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="窜槽检测完整流水线")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small",
                       help="数据规模 (default: small)")
    parser.add_argument("--depth-min", type=float, default=2850.0,
                       help="最小深度 (ft) (default: 2850.0)")
    parser.add_argument("--depth-max", type=float, default=2950.0,
                       help="最大深度 (ft) (default: 2950.0)")
    parser.add_argument("--output-dir", default="outputs",
                       help="输出目录 (default: outputs)")
    
    args = parser.parse_args()
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建并运行流水线
    pipeline = CompletePipeline(
        depth_range=(args.depth_min, args.depth_max),
        data_scale=args.scale,
        output_dir=args.output_dir
    )
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n" + "🎉" * 20)
        print("  窜槽检测完整流水线执行成功!")
        print("🎉" * 20)
        return 0
    else:
        print("\n" + "❌" * 20)
        print("  流水线执行失败，请检查日志")
        print("❌" * 20)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 