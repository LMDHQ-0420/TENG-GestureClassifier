# TENG Gesture Classifier

基于 TENG（摩擦纳米发电机）传感器的手势识别系统。采集 3 通道信号（CH1/CH3/CH5，1000 Hz），通过信号切分、清洗后，提供三种建模方案分类 10 种手势。

## 方法与结果

| 方法 | 信号处理 | 分类器 | Test 准确率 |
|------|---------|--------|------------|
| 基线 | 9 维特征 | 随机森林 | 0.667 |
| **路线A** | VMD 分解 + 111 维多域特征 | ExtraTrees | **0.768** |
| **路线B** | CWT scalogram 时频图 | 2D-CNN | **0.747** |

## 手势类别

| Label | Gesture |
|-------|---------|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |
| 5 | go_the_way |
| 6 | ok |
| 7 | sc |
| 8 | stop |
| 9 | wave |

## 数据集

| 环境 | 目录 | 说明 |
|------|------|------|
| base | `data/raw/base/` | 正常环境（两批次合并，后缀 -1/-2 区分来源） |
| wind_noise | `data/raw/wind_noise/` | 风噪 60-85 dB |
| uv_radiation | `data/raw/uv_radiation/` | 紫外辐照 40mm 30min |

每个 CSV 文件包含多次重复动作（base 约 20 次，uv_radiation 约 5 次），需通过预处理切分为单动作片段。

## 项目结构

```
├── data/
│   ├── raw/                    原始 CSV（3 个环境）
│   └── processed/
│       ├── segments/           切分后的子片段 (.npy)
│       ├── features/           9D 特征向量 (.csv)
│       └── stats/              切分统计报告
├── src/
│   ├── preprocess/             信号准备（共用）
│   │   ├── io.py               CSV 读取 + 标签映射
│   │   ├── segmenter.py        多动作切分（RMS 激活检测）
│   │   ├── cleaner.py          信号清洗（带通+陷波+谱减法）
│   │   ├── features.py         9D 特征提取（基线）
│   │   └── pipeline.py         批处理流水线
│   ├── decompose/             信号分解（路线A/B 共用）
│   │   ├── vmd.py              VMD 变分模态分解
│   │   ├── cwt.py              CWT scalogram 时频图
│   │   └── features_rich.py    111 维多域特征
│   ├── routeA/                路线A：VMD + ExtraTrees
│   │   ├── pipeline_a.py       生成 111 维特征
│   │   ├── train_a.py          训练 + 5折交叉验证
│   │   └── evaluate_a.py
│   ├── routeB/                路线B：CWT + 2D-CNN
│   │   ├── dataset_b.py        scalogram Dataset + 增强
│   │   ├── model_b.py          轻量 2D-CNN
│   │   ├── train_b.py
│   │   └── evaluate_b.py
│   ├── model.py / dataset.py / train.py / evaluate.py   基线随机森林
├── scripts/
│   ├── 00_rename_data.py       数据整理（一次性）
│   └── 01_run_pipeline.py      批量预处理
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb    预处理流程文档
│   ├── 02_sc1_debug.ipynb             切分参数调优案例
│   ├── 03_training_results.ipynb      基线随机森林结果
│   ├── 04_routeA_vmd_features.ipynb   路线A 完整文档
│   └── 05_routeB_cwt_cnn.ipynb        路线B 完整文档
├── PDMA.py                     原始脚本（存档）
├── SNR.py                      原始脚本（存档）
├── code_guide.md               代码逻辑说明
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 创建环境

```bash
conda create -n TENG-GestureClassifier python=3.11
conda activate TENG-GestureClassifier
pip install -r requirements.txt
```

### 2. 数据整理（首次）

```bash
python scripts/00_rename_data.py
```

### 3. 运行预处理

```bash
python scripts/01_run_pipeline.py
```

### 4. 查看预处理效果

```bash
jupyter notebook notebooks/01_preprocessing_demo.ipynb
```

### 5. 训练与评估三种方案

```bash
# 基线：9 维特征 + 随机森林
python -m src.train && python -m src.evaluate

# 路线A：VMD 分解 + 多域特征 + ExtraTrees
python -m src.routeA.pipeline_a    # 生成 111 维特征（首次）
python -m src.routeA.train_a
python -m src.routeA.evaluate_a

# 路线B：CWT scalogram + 2D-CNN
python -m src.routeB.train_b       # 首次自动预计算 scalogram
python -m src.routeB.evaluate_b
```

### 6. 查看可视化文档

```bash
jupyter notebook notebooks/
# 01 预处理 / 02 切分调优 / 03 基线 / 04 路线A / 05 路线B
```

## 预处理流程（三方共用）

```
Raw CSV (多动作/文件)
    ↓ segmenter.py — 滑动 RMS + 跨通道联合检测 + 纯净度自检
单动作片段 (.npy)
    ↓ cleaner.py — 去直流 + 带通(20-450Hz) + 陷波(50Hz) + 谱减法
清洗后片段 ─┬─ 基线: features.py → 9维 → RandomForest
            ├─ 路线A: vmd.py + features_rich.py → 111维 → ExtraTrees
            └─ 路线B: cwt.py → [3,64,128] scalogram → 2D-CNN
```

## 三种建模方案

**基线**：9 维特征（MAV/WL/Ratio）+ 随机森林。简单快速，可解释。

**路线A**：VMD 把每通道分解为 4 个 IMF，从每个 IMF 提取 9 个时域+频域特征（共 111 维），
用 ExtraTrees 分类 + 5 折交叉验证。**当前最优**，频域特征是关键。

**路线B**：CWT 把信号转为 scalogram 时频图（全局 log 归一化保留通道间相对能量），
用轻量 2D-CNN + 强数据增强分类。端到端深度学习，时频图适合论文配图。

## 数据划分

按每个 (环境, 手势) 组合分层抽样，80% 训练 / 20% 测试，每个环境在 train 和 test 中均有样本。
