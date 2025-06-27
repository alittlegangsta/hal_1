#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超声测井数据窜槽严重性分析工具
分析CAST数据中2732-4132ft深度范围内的窜槽分布情况
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from typing import List, Tuple, Dict
import logging
from pathlib import Path

from src.data_loader import DataLoader
from src.plot_config import setup_matplotlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_plotting():
    """设置绘图配置"""
    # 配置matplotlib
    setup_matplotlib()
    
    # 设置中文字体（如果可用）
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [font.name for font in fm.fontManager.ttflist if 
                        any(name in font.name.lower() for name in ['pingfang', 'simsun', 'simhei', 'kaiti'])]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
            print(f"设置中文字体: {chinese_fonts[0]}")
        else:
            print("未发现中文字体，使用英文标签")
    except Exception as e:
        print(f"字体设置失败: {e}")
    
    plt.rcParams['axes.unicode_minus'] = False

class ChannelingAnalyzer:
    """窜槽严重性分析器"""
    
    def __init__(self, depth_range: Tuple[float, float] = (2732.0, 4132.0), 
                 channeling_threshold: float = 2.5):
        """
        初始化分析器
        
        Args:
            depth_range: 分析深度范围 (ft)
            channeling_threshold: 窜槽判断阈值 (Zc < threshold 表示窜槽)
        """
        self.depth_range = depth_range
        self.channeling_threshold = channeling_threshold
        self.data_loader = DataLoader()
        
    def load_and_filter_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载并筛选CAST数据
        
        Returns:
            filtered_cast_data: 筛选后的CAST数据
            filtered_depths: 筛选后的深度数组
            azimuth_angles: 方位角数组
        """
        logger.info("加载CAST数据...")
        
        # 加载数据
        cast_data_dict = self.data_loader.load_cast_data()
        
        # 从字典中提取数据
        all_depths = cast_data_dict['Depth']
        zc_data = cast_data_dict['Zc']  # 形状: (180, 24750)
        
        # 创建方位角数组（0-179度，每2度一个）
        azimuth_angles = np.arange(0, 360, 2)  # 180个方位角
        
        # 筛选深度范围
        depth_mask = (all_depths >= self.depth_range[0]) & (all_depths <= self.depth_range[1])
        filtered_depths = all_depths[depth_mask]
        
        # 筛选Zc数据 - 注意Zc数据是转置的 (180, 24750)，我们需要 (n_depths, 180)
        filtered_zc_data = zc_data[:, depth_mask].T  # 转置为 (n_depths, 180)
        
        logger.info(f"数据筛选完成: 深度点数={len(filtered_depths)}, 方位角数={len(azimuth_angles)}")
        logger.info(f"深度范围: {filtered_depths[0]:.1f} - {filtered_depths[-1]:.1f} ft")
        logger.info(f"Zc数据形状: {filtered_zc_data.shape}")
        
        return filtered_zc_data, filtered_depths, azimuth_angles
        
    def calculate_channeling_ratio(self, cast_data: np.ndarray) -> np.ndarray:
        """
        计算每个深度点的窜槽比例
        
        Args:
            cast_data: CAST数据 (n_depths, n_azimuths)
            
        Returns:
            窜槽比例数组 (n_depths,)
        """
        logger.info("计算窜槽比例...")
        
        # 计算每个深度点窜槽的方位角数量
        channeling_mask = cast_data < self.channeling_threshold
        channeling_counts = np.sum(channeling_mask, axis=1)
        
        # 计算窜槽比例
        total_azimuths = cast_data.shape[1]
        channeling_ratios = channeling_counts / total_azimuths
        
        logger.info(f"窜槽比例计算完成: 平均窜槽比例={np.mean(channeling_ratios):.3f}")
        logger.info(f"最大窜槽比例: {np.max(channeling_ratios):.3f}")
        logger.info(f"最小窜槽比例: {np.min(channeling_ratios):.3f}")
        
        return channeling_ratios
        
    def identify_severe_channeling_zones(self, depths: np.ndarray, 
                                       channeling_ratios: np.ndarray,
                                       severity_threshold: float = 0.3,
                                       min_zone_length: float = 10.0) -> List[Dict]:
        """
        识别窜槽严重区域
        
        Args:
            depths: 深度数组
            channeling_ratios: 窜槽比例数组
            severity_threshold: 严重窜槽阈值 (比例)
            min_zone_length: 最小区域长度 (ft)
            
        Returns:
            严重窜槽区域列表
        """
        logger.info(f"识别严重窜槽区域 (阈值={severity_threshold}, 最小长度={min_zone_length}ft)...")
        
        # 标识严重窜槽点
        severe_mask = channeling_ratios >= severity_threshold
        
        # 查找连续区域
        zones = []
        in_zone = False
        zone_start = None
        
        for i, (depth, is_severe) in enumerate(zip(depths, severe_mask)):
            if is_severe and not in_zone:
                # 开始一个新区域
                in_zone = True
                zone_start = i
            elif not is_severe and in_zone:
                # 结束当前区域
                zone_end = i - 1
                zone_length = depths[zone_end] - depths[zone_start]
                
                if zone_length >= min_zone_length:
                    zone_info = {
                        'start_depth': depths[zone_start],
                        'end_depth': depths[zone_end],
                        'length': zone_length,
                        'avg_channeling_ratio': np.mean(channeling_ratios[zone_start:zone_end+1]),
                        'max_channeling_ratio': np.max(channeling_ratios[zone_start:zone_end+1]),
                        'start_index': zone_start,
                        'end_index': zone_end
                    }
                    zones.append(zone_info)
                
                in_zone = False
        
        # 处理末尾的区域
        if in_zone:
            zone_end = len(depths) - 1
            zone_length = depths[zone_end] - depths[zone_start]
            
            if zone_length >= min_zone_length:
                zone_info = {
                    'start_depth': depths[zone_start],
                    'end_depth': depths[zone_end],
                    'length': zone_length,
                    'avg_channeling_ratio': np.mean(channeling_ratios[zone_start:]),
                    'max_channeling_ratio': np.max(channeling_ratios[zone_start:]),
                    'start_index': zone_start,
                    'end_index': zone_end
                }
                zones.append(zone_info)
        
        # 按严重程度排序
        zones.sort(key=lambda x: x['avg_channeling_ratio'], reverse=True)
        
        logger.info(f"识别到 {len(zones)} 个严重窜槽区域")
        
        return zones
        
    def smooth_channeling_data(self, channeling_ratios: np.ndarray, 
                             window_size: int = 5) -> np.ndarray:
        """
        平滑窜槽数据以减少噪声
        
        Args:
            channeling_ratios: 原始窜槽比例数组
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的窜槽比例数组
        """
        # 使用高斯滤波器平滑数据
        sigma = window_size / 3.0  # 标准差
        smoothed = gaussian_filter1d(channeling_ratios, sigma=sigma)
        return smoothed
        
    def generate_statistics(self, depths: np.ndarray, 
                          channeling_ratios: np.ndarray) -> Dict:
        """
        生成窜槽统计信息
        
        Args:
            depths: 深度数组
            channeling_ratios: 窜槽比例数组
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_depth_range': f"{depths[0]:.1f} - {depths[-1]:.1f} ft",
            'total_length': depths[-1] - depths[0],
            'total_depth_points': len(depths),
            'mean_channeling_ratio': np.mean(channeling_ratios),
            'std_channeling_ratio': np.std(channeling_ratios),
            'median_channeling_ratio': np.median(channeling_ratios),
            'max_channeling_ratio': np.max(channeling_ratios),
            'min_channeling_ratio': np.min(channeling_ratios),
            'severe_points_30pct': np.sum(channeling_ratios >= 0.3),  # 30%以上窜槽
            'severe_points_50pct': np.sum(channeling_ratios >= 0.5),  # 50%以上窜槽
            'severe_points_70pct': np.sum(channeling_ratios >= 0.7),  # 70%以上窜槽
        }
        
        # 计算百分比
        total_points = len(channeling_ratios)
        stats['severe_ratio_30pct'] = stats['severe_points_30pct'] / total_points
        stats['severe_ratio_50pct'] = stats['severe_points_50pct'] / total_points
        stats['severe_ratio_70pct'] = stats['severe_points_70pct'] / total_points
        
        return stats
        
    def plot_comprehensive_analysis(self, depths: np.ndarray, 
                                  channeling_ratios: np.ndarray,
                                  smoothed_ratios: np.ndarray,
                                  severe_zones: List[Dict],
                                  save_path: str = None):
        """
        绘制综合分析图表
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('超声测井数据窜槽严重性分析', fontsize=16, fontweight='bold')
        
        # 图1: 原始和平滑的窜槽比例曲线
        ax1 = axes[0]
        ax1.plot(depths, channeling_ratios, 'lightblue', alpha=0.6, linewidth=0.8, label='原始数据')
        ax1.plot(depths, smoothed_ratios, 'darkblue', linewidth=2, label='平滑数据')
        
        # 标记严重窜槽区域
        for zone in severe_zones:
            ax1.axvspan(zone['start_depth'], zone['end_depth'], 
                       alpha=0.3, color='red', label='严重窜槽区域' if zone == severe_zones[0] else "")
        
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='30%阈值')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%阈值')
        ax1.set_ylabel('窜槽比例')
        ax1.set_title('窜槽比例随深度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 窜槽比例分布直方图
        ax2 = axes[1]
        ax2.hist(channeling_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=np.mean(channeling_ratios), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(channeling_ratios):.3f}')
        ax2.axvline(x=np.median(channeling_ratios), color='green', linestyle='--', 
                   label=f'中位数: {np.median(channeling_ratios):.3f}')
        ax2.set_xlabel('窜槽比例')
        ax2.set_ylabel('频次')
        ax2.set_title('窜槽比例分布直方图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3: 累计窜槽比例
        ax3 = axes[2]
        cumulative_channeling = np.cumsum(channeling_ratios) / np.arange(1, len(channeling_ratios) + 1)
        ax3.plot(depths, cumulative_channeling, 'purple', linewidth=2)
        ax3.set_xlabel('深度 (ft)')
        ax3.set_ylabel('累计平均窜槽比例')
        ax3.set_title('累计平均窜槽比例')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分析图表已保存: {save_path}")
        
        plt.show()
        
    def export_results(self, depths: np.ndarray, 
                      channeling_ratios: np.ndarray,
                      severe_zones: List[Dict],
                      stats: Dict,
                      output_path: str = "channeling_analysis_results.txt"):
        """
        导出分析结果到文件
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("超声测井数据窜槽严重性分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本统计信息
            f.write("基本统计信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"分析深度范围: {stats['total_depth_range']}\n")
            f.write(f"总长度: {stats['total_length']:.1f} ft\n")
            f.write(f"深度点数: {stats['total_depth_points']}\n")
            f.write(f"平均窜槽比例: {stats['mean_channeling_ratio']:.3f}\n")
            f.write(f"标准差: {stats['std_channeling_ratio']:.3f}\n")
            f.write(f"中位数: {stats['median_channeling_ratio']:.3f}\n")
            f.write(f"最大窜槽比例: {stats['max_channeling_ratio']:.3f}\n")
            f.write(f"最小窜槽比例: {stats['min_channeling_ratio']:.3f}\n\n")
            
            # 严重程度分级
            f.write("严重程度分级:\n")
            f.write("-" * 30 + "\n")
            f.write(f"轻度窜槽 (30%以上): {stats['severe_points_30pct']} 点 ({stats['severe_ratio_30pct']:.1%})\n")
            f.write(f"中度窜槽 (50%以上): {stats['severe_points_50pct']} 点 ({stats['severe_ratio_50pct']:.1%})\n")
            f.write(f"重度窜槽 (70%以上): {stats['severe_points_70pct']} 点 ({stats['severe_ratio_70pct']:.1%})\n\n")
            
            # 严重窜槽区域
            f.write("严重窜槽区域 (按严重程度排序):\n")
            f.write("-" * 50 + "\n")
            if severe_zones:
                for i, zone in enumerate(severe_zones, 1):
                    f.write(f"区域 {i}:\n")
                    f.write(f"  深度范围: {zone['start_depth']:.1f} - {zone['end_depth']:.1f} ft\n")
                    f.write(f"  长度: {zone['length']:.1f} ft\n")
                    f.write(f"  平均窜槽比例: {zone['avg_channeling_ratio']:.3f}\n")
                    f.write(f"  最大窜槽比例: {zone['max_channeling_ratio']:.3f}\n")
                    f.write("\n")
            else:
                f.write("未发现符合条件的严重窜槽区域\n\n")
            
            # 建议
            f.write("建议:\n")
            f.write("-" * 30 + "\n")
            if stats['severe_ratio_50pct'] > 0.1:
                f.write("⚠️  发现较多中重度窜槽区域，建议进行水泥胶结修复\n")
            if len(severe_zones) > 0:
                f.write(f"🔧  重点关注 {len(severe_zones)} 个严重窜槽区域\n")
            if stats['mean_channeling_ratio'] > 0.2:
                f.write("📊  整体窜槽比例较高，建议全面评估水泥胶结质量\n")
            
        logger.info(f"分析结果已导出: {output_path}")
        
    def run_complete_analysis(self, severity_threshold: float = 0.3,
                            min_zone_length: float = 10.0,
                            smooth_window: int = 5):
        """
        运行完整的窜槽分析
        
        Args:
            severity_threshold: 严重窜槽阈值
            min_zone_length: 最小区域长度 (ft)
            smooth_window: 平滑窗口大小
        """
        logger.info("开始完整的窜槽严重性分析...")
        
        # 1. 加载数据
        cast_data, depths, azimuth_angles = self.load_and_filter_data()
        
        # 2. 计算窜槽比例
        channeling_ratios = self.calculate_channeling_ratio(cast_data)
        
        # 3. 平滑数据
        smoothed_ratios = self.smooth_channeling_data(channeling_ratios, smooth_window)
        
        # 4. 识别严重区域
        severe_zones = self.identify_severe_channeling_zones(
            depths, smoothed_ratios, severity_threshold, min_zone_length)
        
        # 5. 生成统计信息
        stats = self.generate_statistics(depths, channeling_ratios)
        
        # 6. 打印关键结果
        self.print_key_results(severe_zones, stats)
        
        # 7. 可视化
        save_path = "outputs/figures/channeling_severity_analysis.png"
        Path("outputs/figures").mkdir(parents=True, exist_ok=True)
        self.plot_comprehensive_analysis(depths, channeling_ratios, smoothed_ratios, 
                                       severe_zones, save_path)
        
        # 8. 导出结果
        self.export_results(depths, channeling_ratios, severe_zones, stats)
        
        logger.info("窜槽严重性分析完成！")
        
        return {
            'depths': depths,
            'channeling_ratios': channeling_ratios,
            'smoothed_ratios': smoothed_ratios,
            'severe_zones': severe_zones,
            'statistics': stats
        }
        
    def print_key_results(self, severe_zones: List[Dict], stats: Dict):
        """打印关键分析结果"""
        print("\n" + "="*60)
        print("🔍 窜槽严重性分析结果")
        print("="*60)
        
        print(f"\n📊 基本统计:")
        print(f"   深度范围: {stats['total_depth_range']}")
        print(f"   平均窜槽比例: {stats['mean_channeling_ratio']:.1%}")
        print(f"   最大窜槽比例: {stats['max_channeling_ratio']:.1%}")
        
        print(f"\n⚠️  严重程度分级:")
        print(f"   轻度窜槽 (≥30%): {stats['severe_ratio_30pct']:.1%} 的深度点")
        print(f"   中度窜槽 (≥50%): {stats['severe_ratio_50pct']:.1%} 的深度点")
        print(f"   重度窜槽 (≥70%): {stats['severe_ratio_70pct']:.1%} 的深度点")
        
        print(f"\n🎯 严重窜槽区域 (共{len(severe_zones)}个):")
        if severe_zones:
            for i, zone in enumerate(severe_zones[:5], 1):  # 只显示前5个最严重的
                print(f"   区域{i}: {zone['start_depth']:.1f}-{zone['end_depth']:.1f}ft, "
                      f"长度{zone['length']:.1f}ft, 平均窜槽比例{zone['avg_channeling_ratio']:.1%}")
            if len(severe_zones) > 5:
                print(f"   ... 还有{len(severe_zones)-5}个区域")
        else:
            print("   未发现符合条件的严重窜槽区域")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    # 设置中文字体
    setup_plotting()
    
    print("超声测井数据窜槽严重性分析工具")
    print("="*50)
    
    # 创建分析器
    analyzer = ChannelingAnalyzer(
        depth_range=(2732.0, 4132.0),
        channeling_threshold=2.5
    )
    
    # 运行完整分析
    results = analyzer.run_complete_analysis(
        severity_threshold=0.3,  # 30%窜槽比例作为严重阈值
        min_zone_length=10.0,    # 最小10ft长度的区域
        smooth_window=5          # 5点平滑窗口
    )
    
    print(f"\n✅ 分析完成！")
    print(f"📁 分析图表: outputs/figures/channeling_severity_analysis.png")
    print(f"📄 详细报告: channeling_analysis_results.txt")


if __name__ == "__main__":
    main() 