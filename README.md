# 测井数据窜槽检测项目

## 项目概述

本项目利用高分辨率的超声测井数据（CAST）作为地面实况，通过AI模型分析低分辨率的阵列声波测井数据（XSILMR），精准识别出对水泥胶结窜槽现象最敏感的声波时频特征。该方法具备可逆性，能够将识别出的特征应用于新数据的解释。

## 核心特点

- **高级阵列处理**：应用慢度-时间相干分析等阵列信号处理技术
- **混合特征工程**：结合时频图像特征和物理数值特征
- **深度学习模型**：使用CNN+全连接的混合输入架构
- **模型解释性**：基于Grad-CAM的特征重要性分析
- **方法可逆性**：可应用于新数据的窜槽检测

## 项目结构

```
hal_1/
├── data/
│   └── raw/
│       ├── CAST.mat                    # 超声测井数据
│       └── XSILMR/                     # 声波测井数据
│           ├── XSILMR01.mat - XSILMR13.mat
├── src/                                # 源代码
│   ├── __init__.py
│   ├── data_loader.py                  # 数据加载模块
│   ├── signal_processing.py            # 信号处理模块
│   ├── feature_engineering.py          # 特征工程模块
│   ├── model.py                        # 深度学习模型
│   └── visualization.py                # 可视化和模型解释
├── outputs/                            # 输出结果
│   ├── models/                         # 训练好的模型
│   └── figures/                        # 生成的图表
├── main.py                             # 主程序
├── requirements.txt                    # 依赖包
└── README.md                          # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据说明

### CAST数据（超声测井）
- **文件**: `data/raw/CAST.mat`
- **结构**: 180个方位角 × 24750个深度点
- **参数**: Zc值，< 2.5表示窜槽，≥ 2.5表示胶结良好
- **分辨率**: 高分辨率，每2度一个方位角

### XSILMR数据（声波测井）
- **文件**: `data/raw/XSILMR/XSILMR01.mat` - `XSILMR13.mat`
- **结构**: 13个阵列接收器，每个8个方位接收器
- **数据维度**: 1024个时间点 × 7108个深度点
- **采样**: 时间间隔10μs，接收器间距0.5ft

## 使用方法

### 1. 完整项目流程

```bash
python main.py
```

该命令将执行：
- 数据加载和预处理
- 特征工程（包含阵列信号处理）
- 模型训练
- 模型评估和可视化
- 特征重要性分析
- 可逆性验证

### 2. 新数据应用

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import HybridChannelingModel

# 加载训练好的模型
model = HybridChannelingModel()
model.load_model("outputs/models/channeling_model.h5", 
                 "outputs/models/feature_scaler.pkl")

# 处理新的波形数据
feature_engineer = FeatureEngineer()
scalogram, features = feature_engineer.prepare_new_data(new_waveform)

# 预测窜槽比例
prediction = model.predict_channeling(scalogram, features)
print(f"预测窜槽比例: {prediction[0]:.4f}")
```

## 方法原理

### 1. 数据对齐
- 以第7个阵列接收器为深度基准点
- 计算每个接收器的绝对深度：`D_actual(i) = D_base + (7 - i) × 0.5 ft`
- 筛选目标深度区间：2732-4132 ft

### 2. 信号处理
- **高通滤波**: 4阶巴特沃斯滤波器，截止频率1000Hz
- **慢度-时间相干分析**: 提取局部慢度特征和相干滤波波形
- **衰减率计算**: 计算相邻接收器间的幅度衰减率

### 3. 特征工程
- **图像特征**: 连续小波变换生成的时频尺度图
- **数值特征**: 物理特征（幅值、能量、频率等）+ 慢度 + 衰减率
- **标签生成**: 基于CAST数据计算的窜槽比例（0-1之间）

### 4. 深度学习模型
- **CNN分支**: 处理时频图像特征
- **全连接分支**: 处理数值特征
- **特征融合**: 拼接两个分支的输出
- **回归输出**: Sigmoid激活确保输出在[0,1]范围

### 5. 模型解释
- **Grad-CAM**: 生成特征重要性热力图
- **敏感区域分析**: 识别对窜槽最敏感的时频区域
- **可视化叠加**: 在原始尺度图上高亮重要区域

## 输出结果

### 模型文件
- `channeling_model.h5`: 训练好的深度学习模型
- `feature_scaler.pkl`: 特征标准化器

### 可视化图表
- `data_overview.png`: 数据概览
- `label_distribution.png`: 标签分布
- `training_history.png`: 训练历史
- `prediction_results.png`: 预测结果对比
- `feature_importance.png`: 特征重要性分析
- `gradcam_example.png`: Grad-CAM示例

## 项目优势

1. **科学性强**: 基于物理原理的阵列信号处理
2. **特征丰富**: 结合时频分析和物理特征
3. **模型先进**: 混合输入深度学习架构
4. **解释性好**: Grad-CAM可视化重要特征
5. **可逆性强**: 可直接应用于新数据分析

## 扩展应用

- 实时窜槽监测系统
- 测井质量评估工具
- 井筒完整性分析
- 水泥胶结质量预测

## 注意事项

1. 确保数据文件路径正确
2. 计算资源：建议使用GPU加速训练
3. 内存需求：处理大量数据时需要足够内存
4. 参数调整：可根据具体数据调整模型参数

## 联系信息

如有问题或建议，请联系项目开发者。 