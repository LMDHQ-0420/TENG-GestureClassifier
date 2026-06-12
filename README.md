# TENG 手势识别系统

基于摩擦纳米发电机（TENG）传感器的手势分类系统。传感器采集手部 3 通道电压信号（CH1/CH3/CH5，采样率 1000 Hz），经信号处理和特征提取后，用集成分类器识别 10 种手势。

本项目为材料学课题的配套分析代码，验证 TENG 传感器在不同环境条件（正常、风噪、紫外辐照）下的手势识别能力。

---

## 手势类别

| 标签 | 手势 | 标签 | 手势 |
|------|------|------|------|
| 0 | 1（数字一） | 5 | go_the_way |
| 1 | 2（数字二） | 6 | ok |
| 2 | 3（数字三） | 7 | sc（剪刀手）|
| 3 | 4（数字四） | 8 | stop |
| 4 | 5（数字五） | 9 | wave |

---

## 数据集说明

### 采集环境

| 环境 | 目录 | 条件 | 每文件动作数 |
|------|------|------|------------|
| 正常环境 | `data/raw/base/` | 实验室正常条件 | 约 5~80 次 |
| 风噪环境 | `data/raw/wind_noise/` | 60–85 dB 风噪干扰 | 约 20 次 |
| 紫外辐照 | `data/raw/uv_radiation/` | 距离 40mm 辐照 30 分钟 | 约 5 次 |

### 文件命名规则

base 目录下的文件按采集批次命名，后缀 `-1`、`-2`、`-3` 区分来源：

```
data/raw/base/
├── 1-1.csv       ← 第1批采集，手势"1"
├── 1-2.csv       ← 第2批采集，手势"1"
├── 1-3.csv       ← 第3批采集（新补充），手势"1"
├── ok-1.csv
├── ok-3.csv
└── ...
```

### 原始 CSV 格式

- **无表头，8 列**，逗号分隔
- 使用第 **0、2、4 列**（对应 CH1、CH3、CH5），其余列忽略
- 每行 = 1 个采样点，采样率 1000 Hz
- 每个文件包含同一手势的多次连续重复动作（需切分）

### 数据统计

经预处理切分后（过滤 < 200ms 的极短片段）：

| 环境 | 片段数 |
|------|--------|
| base | 934 |
| wind_noise | 174 |
| uv_radiation | 67 |
| **合计** | **1175** |

---

## 环境配置

### Python 版本

Python 3.11（推荐使用 conda 创建独立环境）

### 创建环境

```bash
conda create -n TENG-GestureClassifier python=3.11
conda activate TENG-GestureClassifier
pip install -r requirements.txt
```

### PyTorch（路线B CNN模型需要，可选）

PyTorch 需根据平台单独安装，不在 requirements.txt 中：

```bash
# Mac（Apple Silicon，MPS 加速）
pip install torch

# Linux/Windows（CPU）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Linux/Windows（CUDA）
# 参考 https://pytorch.org/get-started/locally/
```

### 关键依赖说明

| 包 | 用途 |
|---|---|
| `lightgbm==4.6.0` | 主分类器（最终模型） |
| `vmdpy==0.2` | VMD 变分模态分解（路线A特征提取）|
| `EMD-signal==1.9.0` | EMD 分解（备用，实际使用 VMD）|
| `PyWavelets==1.9.0` | 小波包分解和 CWT（路线B时频图）|
| `xgboost==3.2.0` | Mac 需先安装 `brew install libomp` |
| `umap-learn==0.5.12` | UMAP 可视化（notebook 中使用）|

### Mac 用户注意

XGBoost 依赖 OpenMP，需提前安装：

```bash
brew install libomp
```

---

## 项目结构

```
TENG-GestureClassifier/
│
├── data/
│   ├── raw/                          原始采集数据（不含在 git 中）
│   │   ├── base/                     正常环境（-1/-2/-3 后缀区分批次）
│   │   ├── wind_noise/               风噪环境
│   │   └── uv_radiation/             紫外辐照环境
│   └── processed/                    预处理产物（不含在 git 中）
│       ├── segments/                 切分后的单动作片段 (.npy)
│       ├── features/                 各类特征 CSV 和 NPY 文件
│       └── stats/                    切分统计报告
│
├── src/
│   ├── preprocess/                   信号预处理模块
│   │   ├── io.py                     CSV 读取，手势标签映射
│   │   ├── segmenter.py              多动作切分（滑动RMS+联合检测）
│   │   ├── cleaner.py                信号清洗（带通+陷波+谱减法）
│   │   ├── features.py               基础9维特征（MAV/WL/Ratio）
│   │   └── pipeline.py               批处理流水线
│   │
│   ├── decompose/                    信号分解与特征提取模块
│   │   ├── vmd.py                    VMD 变分模态分解
│   │   ├── cwt.py                    CWT 时频图（scalogram）
│   │   ├── features_enhanced.py      232维增强特征（IMF统计+小波包等）
│   │   ├── features_temporal.py      117维时序特征（时间剖面+包络）
│   │   └── features_rich.py          111维多域特征（路线A使用）
│   │
│   ├── routeA/                       路线A：VMD特征 + 梯度提升
│   │   ├── pipeline_a.py             生成111维特征
│   │   ├── train_a.py                训练 + 5折交叉验证
│   │   └── evaluate_a.py             评估
│   │
│   ├── routeB/                       路线B：CWT时频图 + 2D-CNN
│   │   ├── dataset_b.py              scalogram Dataset + 数据增强
│   │   ├── model_b.py                轻量2D-CNN（26万参数）
│   │   ├── train_b.py                训练
│   │   └── evaluate_b.py             评估
│   │
│   ├── model.py                      最终模型：稳定特征选择 + 4模型软投票
│   ├── train.py                      最终模型训练入口
│   ├── evaluate.py                   最终模型评估入口
│   └── dataset.py                    数据划分工具
│
├── scripts/
│   ├── 00_rename_data.py             数据整理（一次性，将中文目录→英文）
│   └── 01_run_pipeline.py            批量预处理（切分+9维特征）
│
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb   预处理流程完整文档
│   ├── 02_sc1_debug.ipynb            切分参数调优案例（sc手势问题诊断）
│   ├── 03_training_results.ipynb     基线随机森林结果分析
│   ├── 04_routeA_vmd_features.ipynb  路线A：VMD分解+多域特征完整文档
│   ├── 05_routeB_cwt_cnn.ipynb       路线B：CWT时频图+2D-CNN完整文档
│   └── 06_summary_comparison.ipynb   方法汇总对比 + 数据量建议
│
├── checkpoints/                      模型和评估图（不含在 git 中）
├── PDMA.py                           原始信号处理脚本（存档）
├── SNR.py                            原始信噪比分析脚本（存档）
├── code_guide.md                     代码逻辑详细说明
├── requirements.txt
└── README.md
```

---

## 完整使用流程

### 步骤 1：数据整理（首次迁移时运行）

如果原始数据目录是中文命名（基础、风噪60-85db 等），先运行整理脚本：

```bash
python scripts/00_rename_data.py
```

如果已经是 `data/raw/{base,wind_noise,uv_radiation}/` 英文结构，跳过此步。

### 步骤 2：运行预处理流水线

对所有原始 CSV 做信号切分和9维基础特征提取：

```bash
python scripts/01_run_pipeline.py
```

输出：
- `data/processed/segments/{env}/*.npy`：单动作片段
- `data/processed/features/all_features.csv`：9维基础特征
- `data/processed/stats/segment_counts.csv`：切分统计

### 步骤 3：提取最终模型所需特征

首次运行 `src/train.py` 时会自动提取并缓存，也可手动预先生成：

```bash
# 增强特征（232维）—— 需要先运行路线A流水线
python -m src.routeA.pipeline_a

# 时序特征和包络特征在 src/train.py 中自动提取并缓存
```

### 步骤 4：训练最终分类模型

```bash
python -m src.train
```

模型保存到 `checkpoints/ensemble_models.pkl`。

### 步骤 5：评估

```bash
python -m src.evaluate
```

### 查看可视化文档

```bash
conda activate TENG-GestureClassifier
jupyter notebook notebooks/
```

---

## 最终模型说明

### 架构

```
原始片段 [T, 3]
    ↓ 信号清洗（去直流 + 带通20-450Hz + 陷波50Hz + 谱减法）
清洗后片段
    ↓ 特征提取（373维 = 增强232 + 时序117 + 包络24）
    ↓ log 变换（sign(x) * log1p(|x|)）
    ↓ 稳定特征选择（10种子 ExtraTrees 投票，取至少6次选中的 ~96维）
    ↓ 4模型软投票（2×LightGBM + ExtraTrees + SVM）
手势分类（10类）
```

### 性能

全数据集训练 + 全数据集评估（材料课题场景）：

| 指标 | 结果 |
|------|------|
| 整体准确率 | **1.000** |
| 整体 RMSE（概率） | **0.0101** |
| 各手势准确率 | 全部 1.000 |

独立测试集评估（20%留出，seed=42）：

| 指标 | 结果 |
|------|------|
| 整体准确率 | 0.872 |
| 20种子均值 | 0.854 ± 0.017 |
| 排除手势4后准确率 | 0.897 |

### 各环境性能

| 环境 | 准确率 |
|------|--------|
| base（正常） | 0.894 |
| wind_noise（风噪） | 0.765 |
| uv_radiation（辐照） | 0.846 |

---

## 三条技术路线对比

| 路线 | 信号处理 | 模型 | 独立测试准确率 |
|------|---------|------|--------------|
| 基线 | 9维特征 | 随机森林 | 0.667 |
| 路线A | VMD分解 + 111维多域特征 | ExtraTrees | 0.768 |
| 路线B | CWT scalogram 时频图 | 2D-CNN | 0.747 |
| **最终方案** | **373维特征 + log变换** | **4模型软投票** | **0.872** |

---

## 预处理核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 1000 Hz | |
| 使用通道 | CH1, CH3, CH5 | CSV 第 0, 2, 4 列 |
| 带通滤波 | 20–450 Hz，4阶 Butterworth | |
| 陷波滤波 | 50 Hz，Q=30 | 去工频干扰 |
| 激活阈值 | baseline + 3σ | 滑动 RMS 检测 |
| 合并间隔 | 200 ms | 适配 sc 复合手势 |
| VMD 模态数 | K=4 | 低频→高频排列 |
| 最短片段 | 200 ms | 低于此阈值的片段丢弃 |

---

## 迁移注意事项

1. **data/ 目录不在 git 中**，迁移时需单独复制原始数据文件夹
2. **checkpoints/ 中的 .pkl 文件不在 git 中**，迁移后需重新运行训练
3. **data/processed/ 中的 .npy 缓存文件不在 git 中**，迁移后 `src/train.py` 会自动重新生成
4. Mac 用户使用 XGBoost 前需要 `brew install libomp`
5. PyTorch 仅路线B（CWT+CNN）需要，主流程不依赖
6. 建议 Python 3.11，不保证 3.12+ 兼容性（vmdpy 依赖较旧）

---

## 原始脚本（存档）

项目根目录保留了两个原始处理脚本，仅作参考，不在主流程中使用：

- `PDMA.py`：原始信号清洗 + 特征提取（使用 tkinter GUI 选文件）
- `SNR.py`：原始信噪比分析 + 论文配图导出

---

## 版本记录

| 版本 | 内容 |
|------|------|
| v1 | 基础9维特征 + 随机森林，test 0.667 |
| v2 | VMD/CWT双路线，test 0.768/0.747 |
| v3 | 232维增强特征 + 时序特征 + 4模型投票，test 0.870 |
| v4 | + log变换 + 10种子稳定特征选择，test 0.872 |
| **当前** | **全数据训练评估，acc 1.000（材料课题场景）** |
