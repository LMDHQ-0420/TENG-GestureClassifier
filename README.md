# TENG Gesture Classifier

基于 TENG（摩擦纳米发电机）传感器的手势识别系统。采集 3 通道信号（CH1/CH3/CH5，1000 Hz），通过信号切分、清洗、特征提取后，使用随机森林模型分类 10 种手势。

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
│   ├── preprocess/
│   │   ├── io.py               CSV 读取 + 标签映射
│   │   ├── segmenter.py        多动作切分（RMS 激活检测）
│   │   ├── cleaner.py          信号清洗（带通+陷波+谱减法）
│   │   ├── features.py         9D 特征提取
│   │   └── pipeline.py         批处理流水线
│   ├── model.py                随机森林模型
│   ├── dataset.py              数据划分 + 标准化
│   ├── train.py                训练
│   └── evaluate.py             评估 + 混淆矩阵 + 特征重要度
├── scripts/
│   ├── 00_rename_data.py       数据整理（一次性）
│   └── 01_run_pipeline.py      批量预处理
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb    预处理流程文档
│   └── 02_sc1_debug.ipynb             切分参数调优案例
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

### 5. 训练模型

```bash
python -m src.train
```

### 6. 评估模型

```bash
python -m src.evaluate
```

## 预处理流程

```
Raw CSV (多动作/文件)
    ↓ segmenter.py — 滑动 RMS + 跨通道联合检测 + 纯净度自检
单动作片段 (.npy)
    ↓ cleaner.py — 去直流 + 带通(20-450Hz) + 陷波(50Hz) + 谱减法
清洗后片段
    ↓ features.py — [MAV×3, WL×3, Ratio×3]
9D 特征向量
    ↓ model.py — Random Forest (200 trees)
手势分类
```

## 模型

**随机森林**（scikit-learn RandomForestClassifier）

- 200 棵决策树，max_features="sqrt"，class_weight="balanced"
- 输入：9D 标准化特征向量
- 输出：10 类手势

## 数据划分

按每个 (环境, 手势) 组合分层抽样，80% 训练 / 20% 测试，每个环境在 train 和 test 中均有样本。
