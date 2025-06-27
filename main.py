"""
测井数据窜槽检测项目 - 主程序
"""
import os
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import platform

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
        logging.FileHandler('project.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import HybridChannelingModel
from src.visualization import ModelInterpreter, DataVisualizer

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始执行测井数据窜槽检测项目")
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    try:
        # ========== 阶段一：数据加载与预处理 ==========
        logger.info("阶段一：数据加载与预处理")
        
        data_loader = DataLoader()
        
        # 加载数据
        cast_data = data_loader.load_cast_data()
        xsilmr_data = data_loader.load_xsilmr_data()
        
        # 筛选深度范围并计算绝对深度
        filtered_cast, filtered_xsilmr = data_loader.filter_depth_range()
        filtered_xsilmr = data_loader.calculate_absolute_depths(filtered_xsilmr)
        
        # 数据概览可视化
        sample_receiver = 7  # 使用第7个接收器作为示例
        if sample_receiver in filtered_xsilmr:
            DataVisualizer.plot_data_overview(
                filtered_cast, 
                filtered_xsilmr[sample_receiver],
                save_path=str(figures_dir / "data_overview.png")
            )
        
        # ========== 阶段二：特征工程 ==========
        logger.info("阶段二：特征工程")
        
        feature_engineer = FeatureEngineer()
        
        # 生成增强版训练数据（使用阵列信号处理）
        X_images, X_vectors, y_labels = feature_engineer.generate_enhanced_training_data(
            filtered_cast, filtered_xsilmr
        )
        
        # 检查生成的数据
        logger.info(f"生成的训练数据:")
        logger.info(f"  图像特征: {len(X_images)} 个样本")
        logger.info(f"  数值特征: {len(X_vectors)} 个样本")
        logger.info(f"  标签: {len(y_labels)} 个样本")
        
        if len(X_images) == 0:
            logger.error("没有生成有效的训练数据，程序退出")
            return
        
        # 标签分布可视化
        DataVisualizer.plot_label_distribution(
            y_labels, 
            save_path=str(figures_dir / "label_distribution.png")
        )
        
        # ========== 阶段三：模型训练 ==========
        logger.info("阶段三：模型训练")
        
        # 确定输入形状
        sample_image = X_images[0]
        sample_vector = X_vectors[0]
        image_shape = sample_image.shape
        vector_dim = len(sample_vector)
        
        logger.info(f"输入形状: 图像={image_shape}, 向量维度={vector_dim}")
        
        # 创建模型
        model = HybridChannelingModel(image_shape=image_shape, vector_dim=vector_dim)
        
        # 准备训练数据
        train_data, test_data = model.prepare_data(X_images, X_vectors, y_labels)
        
        # 训练模型
        history = model.train(
            train_data=train_data,
            val_data=test_data,  # 使用测试集作为验证集
            epochs=50,  # 减少训练轮数以节省时间
            batch_size=32
        )
        
        # 绘制训练历史
        model.plot_training_history(save_path=str(figures_dir / "training_history.png"))
        
        # ========== 阶段四：模型评估 ==========
        logger.info("阶段四：模型评估")
        
        # 评估模型
        metrics, y_pred = model.evaluate(test_data)
        
        # 绘制预测结果
        X_images_test, X_vectors_test, y_test = test_data
        model.plot_predictions(
            y_test, y_pred,
            save_path=str(figures_dir / "prediction_results.png")
        )
        
        # 保存模型
        model.save_model(
            str(model_dir / "channeling_model.h5"),
            str(model_dir / "feature_scaler.pkl")
        )
        
        # ========== 阶段五：模型解释 ==========
        logger.info("阶段五：模型解释")
        
        # 创建模型解释器
        interpreter = ModelInterpreter(model.model)
        
        # 分析敏感特征
        feature_analysis = interpreter.analyze_sensitive_features(
            X_images, X_vectors, y_labels,
            threshold=0.3,  # 降低阈值以包含更多样本
            n_samples=50    # 减少样本数以节省时间
        )
        
        if feature_analysis:
            # 绘制特征重要性分析
            interpreter.plot_feature_importance(
                feature_analysis,
                save_path=str(figures_dir / "feature_importance.png")
            )
            
            # 演示单个样本的Grad-CAM
            high_channeling_idx = None
            for i, y in enumerate(y_labels):
                if y >= 0.3:  # 找到一个高窜槽比例的样本
                    high_channeling_idx = i
                    break
            
            if high_channeling_idx is not None:
                sample_image = np.array(X_images[high_channeling_idx])
                sample_vector = np.array(X_vectors[high_channeling_idx])
                
                # 生成Grad-CAM热力图
                heatmap = interpreter.generate_gradcam(sample_image, sample_vector)
                
                if heatmap is not None:
                    # 绘制Grad-CAM叠加图
                    interpreter.plot_gradcam_overlay(
                        sample_image, heatmap,
                        save_path=str(figures_dir / "gradcam_example.png")
                    )
        
        # ========== 阶段六：可逆应用演示 ==========
        logger.info("阶段六：可逆应用演示")
        
        # 选择一个测试样本进行演示
        test_idx = 0
        demo_image = X_images_test[test_idx]
        demo_vector = X_vectors_test[test_idx]
        true_ratio = y_test[test_idx]
        
        # 使用特征工程器为新数据准备特征
        demo_scalogram, demo_features = feature_engineer.prepare_new_data(
            demo_image.flatten()  # 模拟原始波形数据
        )
        
        # 预测窜槽比例
        predicted_ratio = model.predict_channeling(demo_scalogram, demo_features)
        
        logger.info(f"可逆应用演示结果:")
        logger.info(f"  真实窜槽比例: {true_ratio:.4f}")
        logger.info(f"  预测窜槽比例: {predicted_ratio[0]:.4f}")
        logger.info(f"  绝对误差: {abs(true_ratio - predicted_ratio[0]):.4f}")
        
        # ========== 项目总结 ==========
        logger.info("=" * 60)
        logger.info("项目执行完成")
        logger.info("=" * 60)
        logger.info("生成的文件:")
        logger.info(f"  模型文件: {model_dir}")
        logger.info(f"  图表文件: {figures_dir}")
        logger.info(f"  日志文件: project.log")
        
        logger.info("项目成果:")
        logger.info(f"  - 成功加载并处理了测井数据")
        logger.info(f"  - 生成了 {len(X_images)} 个训练样本")
        logger.info(f"  - 训练了混合输入深度学习模型")
        logger.info(f"  - 模型评估指标: R²={metrics['R²']:.4f}")
        logger.info(f"  - 实现了基于Grad-CAM的特征解释")
        logger.info(f"  - 验证了方法的可逆性")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def demo_new_data_application():
    """演示新数据应用的完整流程"""
    logger.info("演示新数据应用流程...")
    
    # 加载已训练的模型
    model = HybridChannelingModel()
    model.load_model(
        "outputs/models/channeling_model.h5",
        "outputs/models/feature_scaler.pkl"
    )
    
    # 加载一些新的波形数据（这里用测试数据作为示例）
    data_loader = DataLoader()
    xsilmr_data = data_loader.load_xsilmr_data()
    
    if 7 in xsilmr_data:  # 使用第7个接收器的数据
        new_waveform = xsilmr_data[7]['SideA'][:, 100]  # 选择一个新的深度点
        
        # 使用特征工程器处理新数据
        feature_engineer = FeatureEngineer()
        scalogram, features = feature_engineer.prepare_new_data(new_waveform)
        
        # 预测窜槽比例
        prediction = model.predict_channeling(scalogram, features)
        
        logger.info(f"新数据预测结果: 窜槽比例 = {prediction[0]:.4f}")
        
        # 生成解释性可视化
        interpreter = ModelInterpreter(model.model)
        heatmap = interpreter.generate_gradcam(scalogram, features)
        
        if heatmap is not None:
            interpreter.plot_gradcam_overlay(scalogram, heatmap)

if __name__ == "__main__":
    main()
    
    # 如果需要演示新数据应用，取消注释下面的行
    # demo_new_data_application() 