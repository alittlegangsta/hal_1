#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图配置模块 - 定义标签映射以避免中文字体问题
"""
import matplotlib.pyplot as plt

# 配置matplotlib
def setup_matplotlib():
    """配置matplotlib参数"""
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    print("Matplotlib configured successfully")

# 中英文标签映射
LABELS_MAP = {
    # 通用标签
    '深度': 'Depth',
    '时间': 'Time', 
    '频率': 'Frequency',
    '幅值': 'Amplitude',
    '能量': 'Energy',
    '功率': 'Power',
    '相位': 'Phase',
    
    # 测井相关标签
    '测井数据分析': 'Well Logging Data Analysis',
    '窜槽检测': 'Channeling Detection',
    '声波测井': 'Acoustic Logging',
    '超声测井': 'Ultrasonic Logging',
    '接收器': 'Receiver',
    '方位角': 'Azimuth',
    '深度点': 'Depth Points',
    '窜槽比例': 'Channeling Ratio',
    '胶结质量': 'Cement Quality',
    
    # 信号处理标签
    '原始波形': 'Original Waveform',
    '滤波后波形': 'Filtered Waveform',
    '高通滤波': 'High-pass Filter',
    '连续小波变换': 'Continuous Wavelet Transform',
    '尺度图': 'Scalogram',
    '频谱': 'Spectrum',
    '功率谱': 'Power Spectrum',
    '小波尺度': 'Wavelet Scale',
    '时间采样点': 'Time Samples',
    '慢度': 'Slowness',
    '相干性': 'Coherence',
    '衰减率': 'Attenuation Rate',
    
    # 特征工程标签
    '特征提取': 'Feature Extraction',
    '图像特征': 'Image Features',
    '数值特征': 'Numerical Features',
    '训练样本': 'Training Samples',
    '标签分布': 'Label Distribution',
    '窜槽样本': 'Channeling Samples',
    
    # 模型相关标签
    '模型训练': 'Model Training',
    '训练历史': 'Training History',
    '损失函数': 'Loss Function',
    '验证损失': 'Validation Loss',
    '训练损失': 'Training Loss',
    '预测结果': 'Prediction Results',
    '真实值': 'True Values',
    '预测值': 'Predicted Values',
    '模型评估': 'Model Evaluation',
    
    # 可视化标签
    '数据概览': 'Data Overview',
    '分布直方图': 'Distribution Histogram',
    '特征重要性': 'Feature Importance',
    '热力图': 'Heatmap',
    '叠加图': 'Overlay',
    '重要区域高亮': 'Important Regions Highlighted',
    
    # 单位标签
    '毫秒': 'ms',
    '微秒': 'μs',
    '赫兹': 'Hz',
    '千赫兹': 'kHz',
    '英尺': 'ft',
    '米': 'm',
    '分贝': 'dB',
    
    # 方位标签
    '方位A': 'Side A',
    '方位B': 'Side B',
    '方位C': 'Side C',
    '方位D': 'Side D',
    
    # 统计标签
    '最大值': 'Maximum',
    '最小值': 'Minimum',
    '平均值': 'Mean',
    '中位数': 'Median',
    '标准差': 'Standard Deviation',
    '方差': 'Variance',
    '样本数': 'Sample Count',
    '百分比': 'Percentage',
}

# 专用标签映射函数
def get_label(chinese_label, english_fallback=None):
    """
    获取英文标签
    
    Args:
        chinese_label: 中文标签
        english_fallback: 如果找不到映射，使用的英文备选标签
        
    Returns:
        英文标签
    """
    return LABELS_MAP.get(chinese_label, english_fallback or chinese_label)

# 标题映射
TITLES_MAP = {
    '测井数据窜槽检测项目': 'Well Logging Channeling Detection Project',
    '数据加载和预处理': 'Data Loading and Preprocessing',
    '信号处理': 'Signal Processing',
    '特征工程': 'Feature Engineering',
    '阵列信号处理': 'Array Signal Processing',
    '模型训练与评估': 'Model Training and Evaluation',
    '可视化结果': 'Visualization Results',
    '中文标题测试': 'Chinese Title Test',
    '数据图表测试': 'Data Chart Test',
    '复杂中文文本测试': 'Complex Text Test',
    '热力图测试': 'Heatmap Test',
}

def get_title(chinese_title, english_fallback=None):
    """
    获取英文标题
    
    Args:
        chinese_title: 中文标题
        english_fallback: 如果找不到映射，使用的英文备选标题
        
    Returns:
        英文标题
    """
    return TITLES_MAP.get(chinese_title, english_fallback or chinese_title)

# 初始化matplotlib配置
setup_matplotlib() 