"""
可视化和模型解释模块 - 包含Grad-CAM等解释性方法
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, List, Optional
import logging
import matplotlib.font_manager as fm
import platform

# 配置中文字体
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 简化的字体配置方法
        import matplotlib
        
        # 禁用中文字体，避免警告
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置支持Unicode的后端
        plt.rcParams['text.usetex'] = False
        
        print("字体配置完成，使用英文标签避免中文显示问题")
        
    except Exception as e:
        print(f"字体配置失败: {e}")

# 初始化字体设置
setup_chinese_font()

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """模型解释器"""
    
    def __init__(self, model: keras.Model):
        """
        初始化解释器
        
        Args:
            model: 训练好的Keras模型
        """
        self.model = model
        self.grad_model = None
        self._build_grad_model()
        
    def _build_grad_model(self):
        """构建Grad-CAM所需的梯度模型"""
        if self.model is None:
            logger.warning("模型为空，无法构建梯度模型")
            return
        
        # 找到CNN分支的最后一个卷积层
        conv_layer_name = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                conv_layer_name = layer.name
                break
        
        if conv_layer_name is None:
            logger.warning("未找到卷积层，无法构建Grad-CAM模型")
            return
        
        # 构建梯度模型
        self.grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(conv_layer_name).output, self.model.output]
        )
        
        logger.info(f"Grad-CAM模型构建完成，使用卷积层: {conv_layer_name}")
    
    def generate_gradcam(self, image_input: np.ndarray, vector_input: np.ndarray, 
                        pred_index: Optional[int] = None) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            image_input: 图像输入
            vector_input: 向量输入
            pred_index: 预测索引 (对于回归任务通常为None)
            
        Returns:
            Grad-CAM热力图
        """
        if self.grad_model is None:
            logger.error("Grad-CAM模型未构建成功")
            return None
        
        # 确保输入维度正确
        if image_input.ndim == 2:
            image_input = image_input[np.newaxis, ..., np.newaxis]
        elif image_input.ndim == 3:
            image_input = image_input[np.newaxis, ...]
        
        if vector_input.ndim == 1:
            vector_input = vector_input[np.newaxis, ...]
        
        # 计算梯度
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model([image_input, vector_input])
            
            if pred_index is None:
                # 对于回归任务，使用预测输出
                loss = predictions[0]
            else:
                loss = predictions[:, pred_index]
        
        # 计算卷积层输出相对于损失的梯度
        grads = tape.gradient(loss, conv_outputs)
        
        # 全局平均池化梯度
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 获取卷积层输出和梯度
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        # 计算重要性权重
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # 生成热力图
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # 归一化
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap
    
    def plot_gradcam_overlay(self, scalogram: np.ndarray, heatmap: np.ndarray, 
                           alpha: float = 0.6, save_path: Optional[str] = None):
        """
        绘制Grad-CAM热力图叠加在原始尺度图上
        
        Args:
            scalogram: 原始尺度图
            heatmap: Grad-CAM热力图
            alpha: 透明度
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始尺度图
        im1 = axes[0].imshow(scalogram, aspect='auto', cmap='viridis')
        axes[0].set_title('Original Scalogram')
        axes[0].set_xlabel('Time Samples')
        axes[0].set_ylabel('Wavelet Scale')
        plt.colorbar(im1, ax=axes[0])
        
        # Grad-CAM热力图
        # 将热力图调整到与原始图像相同的尺寸
        heatmap_resized = np.array(tf.image.resize(
            heatmap[..., np.newaxis], 
            scalogram.shape
        ))[:, :, 0]
        
        im2 = axes[1].imshow(heatmap_resized, aspect='auto', cmap='jet', alpha=0.8)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].set_xlabel('Time Samples')
        axes[1].set_ylabel('Wavelet Scale')
        plt.colorbar(im2, ax=axes[1])
        
        # 叠加图
        axes[2].imshow(scalogram, aspect='auto', cmap='gray')
        axes[2].imshow(heatmap_resized, aspect='auto', cmap='jet', alpha=alpha)
        axes[2].set_title('Overlay (Important Regions Highlighted)')
        axes[2].set_xlabel('Time Samples')
        axes[2].set_ylabel('Wavelet Scale')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM可视化保存至: {save_path}")
        
        plt.show()
    
    def analyze_sensitive_features(self, X_images: List, X_vectors: List, 
                                 y_true: List, threshold: float = 0.5, 
                                 n_samples: int = 100) -> Dict:
        """
        分析敏感特征模式
        
        Args:
            X_images: 图像特征列表
            X_vectors: 向量特征列表
            y_true: 真实标签列表
            threshold: 窜槽判断阈值
            n_samples: 分析样本数量
            
        Returns:
            敏感特征分析结果
        """
        logger.info("分析敏感特征模式...")
        
        # 筛选高窜槽比例的样本
        high_channeling_indices = [i for i, y in enumerate(y_true) if y >= threshold]
        
        if len(high_channeling_indices) == 0:
            logger.warning("没有找到高窜槽比例的样本")
            return {}
        
        # 随机选择样本进行分析
        selected_indices = np.random.choice(
            high_channeling_indices, 
            min(n_samples, len(high_channeling_indices)), 
            replace=False
        )
        
        # 收集热力图
        heatmaps = []
        for idx in selected_indices:
            image = np.array(X_images[idx])
            vector = np.array(X_vectors[idx])
            
            heatmap = self.generate_gradcam(image, vector)
            if heatmap is not None:
                heatmaps.append(heatmap)
        
        if len(heatmaps) == 0:
            logger.warning("未能生成有效的热力图")
            return {}
        
        # 计算平均热力图
        avg_heatmap = np.mean(heatmaps, axis=0)
        
        # 找到最敏感的区域
        threshold_heat = np.percentile(avg_heatmap, 90)  # 前10%的敏感区域
        sensitive_regions = avg_heatmap >= threshold_heat
        
        # 分析敏感区域的特征
        sensitive_scales, sensitive_times = np.where(sensitive_regions)
        
        results = {
            'avg_heatmap': avg_heatmap,
            'sensitive_regions': sensitive_regions,
            'sensitive_scales': sensitive_scales,
            'sensitive_times': sensitive_times,
            'n_analyzed_samples': len(heatmaps)
        }
        
        logger.info(f"敏感特征分析完成: 分析了{len(heatmaps)}个样本")
        
        return results
    
    def plot_feature_importance(self, feature_analysis: Dict, 
                              save_path: Optional[str] = None):
        """
        绘制特征重要性分析结果
        
        Args:
            feature_analysis: 特征分析结果
            save_path: 保存路径
        """
        if not feature_analysis:
            logger.warning("特征分析结果为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 平均热力图
        im1 = axes[0, 0].imshow(feature_analysis['avg_heatmap'], 
                               aspect='auto', cmap='hot')
        axes[0, 0].set_title('Average Sensitivity Heatmap')
        axes[0, 0].set_xlabel('Time Samples')
        axes[0, 0].set_ylabel('Wavelet Scale')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 敏感区域
        axes[0, 1].imshow(feature_analysis['sensitive_regions'], 
                         aspect='auto', cmap='binary')
        axes[0, 1].set_title('Most Sensitive Regions (Top 10%)')
        axes[0, 1].set_xlabel('Time Samples')
        axes[0, 1].set_ylabel('Wavelet Scale')
        
        # 尺度维度的敏感性分布
        scale_sensitivity = np.mean(feature_analysis['avg_heatmap'], axis=1)
        axes[1, 0].plot(scale_sensitivity)
        axes[1, 0].set_title('Scale Dimension Sensitivity')
        axes[1, 0].set_xlabel('Wavelet Scale Index')
        axes[1, 0].set_ylabel('Average Sensitivity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 时间维度的敏感性分布
        time_sensitivity = np.mean(feature_analysis['avg_heatmap'], axis=0)
        axes[1, 1].plot(time_sensitivity)
        axes[1, 1].set_title('Time Dimension Sensitivity')
        axes[1, 1].set_xlabel('Time Samples')
        axes[1, 1].set_ylabel('Average Sensitivity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性分析图保存至: {save_path}")
        
        plt.show()

class DataVisualizer:
    """数据可视化器"""
    
    @staticmethod
    def plot_data_analysis(images: np.ndarray, vectors: np.ndarray, 
                          labels: np.ndarray, save_path: Optional[str] = None):
        """
        绘制数据分析图表
        
        Args:
            images: 图像特征数组
            vectors: 向量特征数组  
            labels: 标签数组
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 标签分布直方图
        axes[0, 0].hist(labels, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].set_title('Channeling Ratio Distribution')
        axes[0, 0].set_xlabel('Channeling Ratio')
        axes[0, 0].set_ylabel('Sample Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 图像特征统计 - 最大值分布
        image_max_vals = np.max(images.reshape(images.shape[0], -1), axis=1)
        axes[0, 1].hist(image_max_vals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Image Feature Max Values')
        axes[0, 1].set_xlabel('Max Value')
        axes[0, 1].set_ylabel('Sample Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 向量特征相关性热力图
        import seaborn as sns
        # 处理零方差特征避免NaN值
        vectors_for_corr = vectors.copy()
        for i in range(vectors.shape[1]):
            if np.std(vectors[:, i]) == 0:
                vectors_for_corr[:, i] += np.random.normal(0, 1e-8, vectors.shape[0])
        
        vector_corr = np.corrcoef(vectors_for_corr.T)
        # 处理NaN值
        vector_corr = np.nan_to_num(vector_corr, nan=0.0)
        
        feature_names = ['Max Amp', 'RMS Amp', 'Energy', 'Zero Cross', 
                        'Dom Freq', 'Spect Cent', 'Receiver', 'Azimuth']
        sns.heatmap(vector_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=feature_names, yticklabels=feature_names,
                   ax=axes[0, 2], fmt='.2f')
        axes[0, 2].set_title('Vector Features Correlation')
        
        # 标签vs主要特征的散点图
        axes[1, 0].scatter(vectors[:, 0], labels, alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Max Amplitude')
        axes[1, 0].set_ylabel('Channeling Ratio')
        axes[1, 0].set_title('Max Amplitude vs Channeling Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 能量特征vs标签
        axes[1, 1].scatter(vectors[:, 2], labels, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Energy')
        axes[1, 1].set_ylabel('Channeling Ratio')
        axes[1, 1].set_title('Energy vs Channeling Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 样本图像示例 - 显示一个代表性样本
        # 选择中位数样本
        mid_idx = np.argmin(np.abs(labels - np.median(labels)))
        sample_image = images[mid_idx]
        sample_label = labels[mid_idx]
        
        # 在最后一个子图显示样本图像
        im = axes[1, 2].imshow(sample_image, aspect='auto', cmap='viridis')
        axes[1, 2].set_title(f'Sample Scalogram (ratio={sample_label:.3f})')
        axes[1, 2].set_xlabel('Time Samples')
        axes[1, 2].set_ylabel('Wavelet Scale')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"数据分析图表保存至: {save_path}")
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_data_overview(cast_data: Dict, xsilmr_sample: Dict, 
                          save_path: Optional[str] = None):
        """
        绘制数据概览
        
        Args:
            cast_data: CAST数据
            xsilmr_sample: XSILMR样本数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CAST数据 - Zc分布
        zc_flat = cast_data['Zc'].flatten()
        axes[0, 0].hist(zc_flat, bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(x=2.5, color='red', linestyle='--', 
                          label='Channeling Threshold (2.5)')
        axes[0, 0].set_title('CAST Zc Value Distribution')
        axes[0, 0].set_xlabel('Zc Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CAST数据 - 深度-方位图
        im1 = axes[0, 1].imshow(cast_data['Zc'], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('CAST Data (Azimuth-Depth)')
        axes[0, 1].set_xlabel('Depth Point Index')
        axes[0, 1].set_ylabel('Azimuth Index')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # XSILMR样本波形
        if 'SideA' in xsilmr_sample:
            waveform_sample = xsilmr_sample['SideA'][:, 0]  # 第一个深度点的波形
            time_axis = np.arange(len(waveform_sample)) * 1e-5 * 1000  # 转换为毫秒
            axes[1, 0].plot(time_axis, waveform_sample)
            axes[1, 0].set_title('XSILMR Waveform Example')
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Amplitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # XSILMR样本尺度图
        if 'SideA' in xsilmr_sample:
            # 简单的时频表示
            from scipy import signal
            f, t, Sxx = signal.spectrogram(waveform_sample, fs=1e5, nperseg=64)
            im2 = axes[1, 1].pcolormesh(t*1000, f/1000, 10*np.log10(Sxx), 
                                       shading='gouraud')
            axes[1, 1].set_title('XSILMR Spectrum Example')
            axes[1, 1].set_xlabel('Time (ms)')
            axes[1, 1].set_ylabel('Frequency (kHz)')
            plt.colorbar(im2, ax=axes[1, 1], label='Power (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"数据概览图保存至: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_label_distribution(y_labels: List, save_path: Optional[str] = None):
        """
        绘制标签分布
        
        Args:
            y_labels: 标签列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        y_array = np.array(y_labels)
        
        # 直方图
        axes[0].hist(y_array, bins=50, alpha=0.7, color='green')
        axes[0].set_title('Channeling Ratio Distribution')
        axes[0].set_xlabel('Channeling Ratio')
        axes[0].set_ylabel('Sample Count')
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_labels = np.sort(y_array)
        p = np.arange(len(sorted_labels)) / len(sorted_labels)
        axes[1].plot(sorted_labels, p)
        axes[1].set_title('Channeling Ratio Cumulative Distribution')
        axes[1].set_xlabel('Channeling Ratio')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"标签分布图保存至: {save_path}")
        
        plt.show() 