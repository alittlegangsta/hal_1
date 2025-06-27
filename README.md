# 窜槽检测完整流水线

一键运行从数据处理到模型训练到结果输出的完整流程，基于HDF5增量存储技术，解决内存不足问题。

## 🚀 快速开始

### 一键运行完整流水线

```bash
# 快速测试 (推荐初次使用)
python run_complete_pipeline.py --scale small

# 中等规模训练
python run_complete_pipeline.py --scale medium

# 完整规模训练
python run_complete_pipeline.py --scale large
```

### 高级用法

```bash
# 自定义深度范围和输出目录
python run_complete_pipeline.py \
    --scale medium \
    --depth-min 2800.0 \
    --depth-max 3000.0 \
    --output-dir my_results

# 查看帮助
python run_complete_pipeline.py --help
```

## 📊 数据规模说明

| 规模 | 接收器 | 方位角扇区 | 深度点/扇区 | 预估样本数 | 预估时间 | 用途 |
|------|--------|------------|-------------|------------|----------|------|
| **small** | 1个 | 2个 | 50个 | ~100 | 2-5分钟 | 快速测试验证 |
| **medium** | 5个 | 4个 | 100个 | ~2,000 | 20-40分钟 | 日常开发训练 |
| **large** | 13个 | 8个 | 全部(~200) | ~20,000 | 2-4小时 | 生产级训练 |

## 🎯 核心特性

### ✅ 内存高效处理
- **HDF5增量存储** - 数据直接写入硬盘
- **批处理机制** - 分块处理，避免内存溢出
- **实时垃圾回收** - 自动清理内存
- **精确空间预分配** - 无零值样本问题

### ✅ 完整流水线
1. **数据处理阶段** - HDF5增量存储
2. **数据分析阶段** - 统计分析和可视化
3. **模型训练阶段** - 深度学习训练
4. **结果输出阶段** - 评估报告和图表

### ✅ 自动化输出
- 📊 **数据分析图表** - 数据分布和质量分析
- 📈 **训练历史图** - 损失和指标变化
- 🎯 **预测结果图** - 真实值vs预测值对比
- 📋 **完整报告** - 详细的执行报告
- 💾 **模型文件** - 训练好的模型和标准化器

## 📁 输出结构

运行完成后，会在输出目录生成以下文件：

```
outputs/
├── data/                           # 数据文件
│   └── features_small_2850_2950.h5   # HDF5特征数据
├── models/                         # 模型文件
│   ├── channeling_model_small.h5     # 训练好的模型
│   └── scaler_small.pkl              # 特征标准化器
├── figures/                        # 图表文件
│   ├── data_analysis_small.png        # 数据分析图
│   ├── training_history_small.png     # 训练历史图
│   └── predictions_small.png          # 预测结果图
├── logs/                           # 日志文件
│   └── pipeline_20231227_143022.log   # 详细执行日志
└── pipeline_report_small.txt       # 完整执行报告
```

## 🔧 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 系统要求

- Python 3.8+
- 内存：最少4GB，推荐8GB+
- 硬盘：最少10GB可用空间
- macOS/Linux/Windows

## 📈 性能优化

### 内存不足问题 ✅ 已解决
- **问题**：传统方法将所有数据加载到内存，导致内存不足
- **解决**：HDF5增量存储，数据直接写入硬盘
- **效果**：内存使用减少90%+，支持任意规模数据

### 零值样本问题 ✅ 已修复
- **问题**：HDF5预分配空间过大，产生大量零值样本
- **解决**：精确估算样本数，按需分配空间
- **效果**：零值样本从92.6%降到0%

## 🛠️ 技术架构

### 数据处理流程
```
原始数据 → 深度筛选 → 信号处理 → 特征工程 → HDF5存储
```

### 模型架构
```
CNN图像分支 + 全连接数值分支 → 特征融合 → 窜槽比例预测
```

### HDF5增量存储
```
批处理缓冲区 → 定期写入硬盘 → 内存清理 → 垃圾回收
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 使用更小的批处理大小
   python run_complete_pipeline.py --scale small
   ```

2. **磁盘空间不足**
   ```bash
   # 指定其他输出目录
   python run_complete_pipeline.py --output-dir /path/to/large/disk
   ```

3. **数据文件不存在**
   ```
   确保 data/raw/ 目录包含：
   - CAST.mat
   - XSILMR/*.mat (XSILMR01.mat 到 XSILMR13.mat)
   ```

### 日志查看

```bash
# 查看最新日志
tail -f outputs/logs/pipeline_*.log

# 查看完整报告
cat outputs/pipeline_report_*.txt
```

## 📝 开发说明

### 项目结构

```
.
├── run_complete_pipeline.py    # 🚀 主要脚本：完整流水线
├── main.py                     # 原始主程序（保留兼容性）
├── src/                        # 源代码模块
│   ├── data_loader.py          # 数据加载器
│   ├── signal_processing.py    # 信号处理
│   ├── feature_engineering.py  # 特征工程
│   ├── hdf5_manager.py         # HDF5数据管理
│   ├── model.py                # 深度学习模型
│   └── visualization.py        # 数据可视化
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
└── outputs/                    # 输出目录
```

### 自定义扩展

如需自定义处理流程，可以：

1. **修改数据规模配置**：编辑 `CompletePipeline.scale_config`
2. **调整模型参数**：修改 `src/model.py` 中的模型结构
3. **自定义可视化**：扩展 `src/visualization.py` 的图表功能

## 📜 版本历史

### v2.0 - 完整流水线版本
- ✅ 一站式完整流水线脚本
- ✅ HDF5增量存储解决内存问题
- ✅ 零值样本问题修复
- ✅ 自动化结果输出
- ✅ 多种数据规模支持

### v1.0 - 基础版本
- 基础的数据处理和模型训练功能

## 📞 支持

如有问题，请查看：
1. 详细日志：`outputs/logs/pipeline_*.log`
2. 执行报告：`outputs/pipeline_report_*.txt`
3. 故障排除章节 