"""
测井数据窜槽检测项目 - 简化演示脚本
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.signal_processing import SignalProcessor
from src.visualization import DataVisualizer

def demo_data_loading():
    """演示数据加载功能"""
    print("=" * 50)
    print("演示1: 数据加载和预处理")
    print("=" * 50)
    
    # 创建数据加载器
    loader = DataLoader()
    
    # 加载数据
    cast_data = loader.load_cast_data()
    xsilmr_data = loader.load_xsilmr_data()
    
    # 筛选深度范围
    filtered_cast, filtered_xsilmr = loader.filter_depth_range()
    filtered_xsilmr = loader.calculate_absolute_depths(filtered_xsilmr)
    
    print(f"CAST数据: {cast_data['Depth'].shape[0]} 个深度点, {cast_data['Zc'].shape[0]} 个方位角")
    print(f"XSILMR数据: {len(xsilmr_data)} 个接收器")
    print(f"筛选后CAST数据: {len(filtered_cast['Depth'])} 个深度点")
    print(f"筛选后XSILMR数据: {len(filtered_xsilmr[7]['Depth'])} 个深度点")
    
    # 分析窜槽分布
    zc_values = filtered_cast['Zc'].flatten()
    channeling_ratio = np.sum(zc_values < 2.5) / len(zc_values)
    print(f"总体窜槽比例: {channeling_ratio:.2%}")
    
    return filtered_cast, filtered_xsilmr

def demo_signal_processing(xsilmr_data):
    """演示信号处理功能"""
    print("\n" + "=" * 50)
    print("演示2: 信号处理")
    print("=" * 50)
    
    # 创建信号处理器
    processor = SignalProcessor()
    
    # 选择一个接收器的数据进行演示
    receiver_idx = 7
    receiver_data = xsilmr_data[receiver_idx]
    
    # 获取原始波形（选择方位A的第一个深度点）
    original_waveform = receiver_data['SideA'][:, 0]
    print(f"原始波形长度: {len(original_waveform)}")
    print(f"原始波形最大幅值: {np.max(np.abs(original_waveform)):.2e}")
    
    # 应用高通滤波
    filtered_waveform = processor.apply_highpass_filter(original_waveform)
    print(f"滤波后最大幅值: {np.max(np.abs(filtered_waveform)):.2e}")
    
    # 生成尺度图
    scalogram = processor.generate_scalogram(filtered_waveform)
    print(f"尺度图形状: {scalogram.shape}")
    
    # 提取物理特征
    features = processor.extract_physical_features(filtered_waveform)
    print("提取的物理特征:")
    for key, value in features.items():
        print(f"  {key}: {value:.2e}")
    
    # 创建一个简单的可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原始波形
    time_axis = np.arange(len(original_waveform)) * 1e-5 * 1000  # 转换为毫秒
    axes[0, 0].plot(time_axis, original_waveform)
    axes[0, 0].set_title('Original Waveform')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 滤波后波形
    axes[0, 1].plot(time_axis, filtered_waveform)
    axes[0, 1].set_title('High-pass Filtered Waveform')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 尺度图
    im = axes[1, 0].imshow(scalogram, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Continuous Wavelet Transform Scalogram')
    axes[1, 0].set_xlabel('Time Samples')
    axes[1, 0].set_ylabel('Wavelet Scale')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 频谱
    fft = np.fft.fft(filtered_waveform)
    freqs = np.fft.fftfreq(len(filtered_waveform), 1e-5)
    power_spectrum = np.abs(fft)**2
    
    # 只显示正频率部分
    pos_freqs = freqs[:len(freqs)//2]
    pos_power = power_spectrum[:len(power_spectrum)//2]
    
    axes[1, 1].plot(pos_freqs/1000, 10*np.log10(pos_power + 1e-10))
    axes[1, 1].set_title('Power Spectrum')
    axes[1, 1].set_xlabel('Frequency (kHz)')
    axes[1, 1].set_ylabel('Power (dB)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 50)  # 只显示0-50kHz
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "signal_processing_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return scalogram, features

def demo_feature_engineering(cast_data, xsilmr_data):
    """演示特征工程功能"""
    print("\n" + "=" * 50)
    print("演示3: 特征工程")
    print("=" * 50)
    
    # 创建特征工程器
    feature_engineer = FeatureEngineer()
    
    # 生成少量训练数据进行演示（限制数据量以节省时间）
    print("生成训练数据样本...")
    
    # 只使用一个接收器和一个方位的少量数据
    demo_xsilmr = {7: xsilmr_data[7]}  # 只使用第7个接收器
    
    # 限制深度点数量
    n_samples = min(50, len(demo_xsilmr[7]['Depth']))  # 只使用前50个深度点或者所有可用的深度点
    demo_xsilmr[7]['Depth'] = demo_xsilmr[7]['Depth'][:n_samples]
    demo_xsilmr[7]['AbsoluteDepth'] = demo_xsilmr[7]['AbsoluteDepth'][:n_samples]
    demo_xsilmr[7]['SideA'] = demo_xsilmr[7]['SideA'][:, :n_samples]
    
    X_images, X_vectors, y_labels = feature_engineer.generate_training_data(
        cast_data, demo_xsilmr
    )
    
    print(f"生成的样本数: {len(X_images)}")
    if len(X_images) > 0:
        print(f"图像特征形状: {X_images[0].shape}")
        print(f"数值特征维度: {len(X_vectors[0])}")
        print(f"标签范围: {np.min(y_labels):.3f} - {np.max(y_labels):.3f}")
        
        # 分析标签分布
        channeling_samples = np.sum(np.array(y_labels) > 0.5)
        print(f"高窜槽样本数 (>0.5): {channeling_samples}/{len(y_labels)}")
        
        # 可视化标签分布
        DataVisualizer.plot_label_distribution(y_labels)
        
        return X_images, X_vectors, y_labels
    else:
        print("未生成有效的训练样本")
        return [], [], []

def demo_array_processing(xsilmr_data):
    """演示阵列信号处理功能"""
    print("\n" + "=" * 50)
    print("演示4: 阵列信号处理")
    print("=" * 50)
    
    # 创建信号处理器
    processor = SignalProcessor()
    
    # 收集某个方位的所有接收器数据
    side = 'A'
    side_waveforms = {}
    
    for receiver_idx in sorted(xsilmr_data.keys()):
        if f'Side{side}' in xsilmr_data[receiver_idx]:
            # 只使用前10个深度点以节省计算时间
            waveforms = xsilmr_data[receiver_idx][f'Side{side}'][:, :10]
            filtered_waveforms = processor.apply_highpass_filter(waveforms)
            side_waveforms[receiver_idx] = filtered_waveforms
    
    print(f"收集到 {len(side_waveforms)} 个接收器的方位{side}数据")
    
    # 执行慢度-时间相干性分析
    print("执行慢度-时间相干性分析...")
    local_slowness, coherent_waveforms, quality_metrics = processor.slowness_time_coherence(side_waveforms)
    
    print(f"局部慢度数组形状: {local_slowness.shape}")
    print(f"相干滤波后波形数: {len(coherent_waveforms)}")
    print(f"慢度计算质量 - 平均R²值: {np.mean(quality_metrics):.3f}")
    print(f"慢度计算质量 - 高质量比例 (R²>0.5): {np.sum(quality_metrics > 0.5)/quality_metrics.size:.1%}")
    
    # 计算衰减率
    print("计算衰减率...")
    attenuation = processor.calculate_attenuation(coherent_waveforms)
    print(f"衰减率数组形状: {attenuation.shape}")
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原始波形（第一个深度点）
    for i, (receiver_id, waveform) in enumerate(list(side_waveforms.items())[:5]):
        axes[0, 0].plot(waveform[:, 0], label=f'Receiver {receiver_id}', alpha=0.7)
    axes[0, 0].set_title('Original Waveform Comparison (1st depth)')
    axes[0, 0].set_xlabel('Time Samples')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 相干滤波后波形
    for i, (receiver_id, waveform) in enumerate(list(coherent_waveforms.items())[:5]):
        axes[0, 1].plot(waveform[:, 0], label=f'Receiver {receiver_id}', alpha=0.7)
    axes[0, 1].set_title('Coherent Filtered Waveform (1st depth)')
    axes[0, 1].set_xlabel('Time Samples')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 局部慢度
    im1 = axes[1, 0].imshow(local_slowness, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Local Slowness')
    axes[1, 0].set_xlabel('Depth Points')
    axes[1, 0].set_ylabel('Receiver Index')
    plt.colorbar(im1, ax=axes[1, 0], label='Slowness')
    
    # 衰减率
    im2 = axes[1, 1].imshow(attenuation, aspect='auto', cmap='plasma')
    axes[1, 1].set_title('Attenuation Rate')
    axes[1, 1].set_xlabel('Depth Points')
    axes[1, 1].set_ylabel('Receiver Spacing Index')
    plt.colorbar(im2, ax=axes[1, 1], label='Attenuation')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "array_processing_demo.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主演示函数"""
    print("测井数据窜槽检测项目 - 功能演示")
    print("=" * 60)
    
    try:
        # 演示1: 数据加载
        filtered_cast, filtered_xsilmr = demo_data_loading()
        
        # 演示2: 信号处理
        scalogram, features = demo_signal_processing(filtered_xsilmr)
        
        # 演示3: 特征工程
        X_images, X_vectors, y_labels = demo_feature_engineering(filtered_cast, filtered_xsilmr)
        
        # 演示4: 阵列信号处理
        demo_array_processing(filtered_xsilmr)
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("主要成果:")
        print(f"- 成功加载了 {len(filtered_cast['Depth'])} 个CAST深度点")
        print(f"- 成功加载了 {len(filtered_xsilmr)} 个XSILMR接收器")
        print(f"- 演示了信号处理流程（滤波、小波变换等）")
        print(f"- 生成了 {len(X_images)} 个训练样本")
        print(f"- 演示了阵列信号处理功能")
        print("- 所有演示图表已保存到 outputs/figures/ 目录")
        
        # 给出下一步建议
        print("\n下一步可以:")
        print("1. 运行 python main.py 执行完整的训练流程")
        print("2. 调整参数优化模型性能")
        print("3. 添加更多的特征工程方法")
        print("4. 扩展到其他测井应用场景")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 