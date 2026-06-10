# TENG-GestureClassifier 代码处理逻辑说明

> 本文件记录所有数据处理与模型代码的逻辑，每次修改代码必须同步更新此文件。

---

## 数据背景

| 参数 | 值 |
|------|-----|
| 采样率 | 1000 Hz |
| 使用通道 | CH1、CH3、CH5（CSV 第 0、2、4 列，0-indexed） |
| CSV 格式 | 无表头，8 列原始数值 |
| 手势类别 | 1, 2, 3, 4, 5, go_the_way, ok, sc, stop, wave（共 10 类） |

| 数据集 | 目录 | 每文件动作数 | 说明 |
|--------|------|-------------|------|
| base | data/raw/base/ | -1 后缀约 20 次，-2 后缀约 5 次 | 正常环境（两批次合并） |
| wind_noise | data/raw/wind_noise/ | 约 20 次 | 风噪 60-85 dB |
| uv_radiation | data/raw/uv_radiation/ | 约 5 次 | 紫外辐照 40mm 30min |

---

## 预处理流程（src/preprocess/）

### 总体流水线（pipeline.py）

```
Raw CSV → segmenter → cleaner → features → 9D 特征 CSV
```

调用入口：`run_all(raw_root, processed_root)` 或 `scripts/01_run_pipeline.py`

### 1. 文件读取（io.py）

- `load_raw_csv(path)` → 读取无表头 CSV，提取第 0/2/4 列，返回 `[N, 3]` float64 数组
- `get_gesture_name(filename)` → 从文件名提取手势名（去掉 `-1`/`-2` 来源后缀）
- `get_gesture_label(filename)` → 手势名映射到整数标签 0-9
- 多编码自动探测（utf-8-sig, utf-8, gbk, gb18030, latin1）

### 2. 多动作切分（segmenter.py）

**来源**：SNR.py 的 `analyze_channel` + `choose_representative_joint_action`

**关键改动**：SNR.py 原来只选一个代表动作（中位 SNR），重构后返回所有通过纯净度自检的联合动作段。

#### 2.1 单通道激活检测（`analyze_single_channel`）

1. 以中位数去中心化
2. 滑动 RMS（200 ms 窗口）→ `rms_env`
3. 基线估计：取 `rms_env` 最低 20% 的均值/标准差
4. 激活阈值：`baseline_mean + 3 × baseline_std`
5. 检测激活段：最短 100 ms，合并 <50 ms 的碎片间隙

#### 2.2 跨通道联合动作检测（`detect_joint_actions`）

1. 统计每时刻激活通道数 `active_count`
2. 至少 2 通道同时激活 → 核心段
3. 向两侧扩展至多 250 ms，合并相邻核心段（间隔 <80 ms）
4. 单个动作最长 2.8 s
5. **纯净度自检**：任一通道在窗口内出现多次激活 → 丢弃

#### 2.3 切分输出（`segment_file`）

- 对每个联合动作段两侧各加 50 ms padding
- 返回 `List[np.ndarray]`，每个元素 shape `[T_i, 3]`
- 保存为 `.npy` 文件到 `data/processed/segments/{env}/`

#### 参数表（`SegParams`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rms_window_s | 0.200 | 滑动 RMS 窗口 |
| min_active_s | 0.100 | 最短激活持续时间 |
| merge_gap_s | 0.200 | 单通道碎片合并间隔（从 0.050 调高，适配 sc 等复合手势） |
| baseline_low_percentile | 0.20 | 基线估计分位数 |
| threshold_k | 3.0 | 阈值系数 |
| joint_merge_gap_s | 0.080 | 跨通道核心段合并间隔 |
| joint_edge_extension_s | 0.250 | 联合动作扩展量 |
| max_joint_active_s | 2.800 | 单动作最长时间 |
| joint_min_active_channels | 2 | 联合动作最少激活通道数 |
| pre_post_pad_s | 0.050 | 切分两侧额外保留 |

### 3. 单片段信号清洗（cleaner.py）

**来源**：PDMA.py 的滤波流程

对每个切出的子片段（单次动作）依次执行：

1. **去直流**：`col -= mean(col)`
2. **带通滤波**：4 阶 Butterworth 20–450 Hz
3. **陷波滤波**：IIR Notch 50 Hz，Q=30
4. **谱减法降噪**（`spectral_subtraction`）：
   - STFT（nperseg=256）→ 取最低 10% 能量帧作噪声估计
   - `subtracted_mag = max(|X|² − 2·noise_mu², (0.02·noise_mu)²)^0.5`
   - ISTFT 重建
5. **边界裁切**（`detect_artifact_boundary`）：
   - 首尾各 500 ms 搜索范围内找包络最小值点
   - 三通道联合取最保守边界

### 4. 9D 特征提取（features.py）

**来源**：PDMA.py 的 `calculate_9d_vector`

| 特征 | 含义 | 维度 |
|------|------|------|
| MAV | 平均绝对值 `mean(\|x\|)` | 3 |
| WL | 波形长度 `sum(\|diff(x)\|) / duration` | 3 |
| Ratio | 各通道 RMS 占总 RMS 之比 | 3 |

输出：`[MAV_CH1, MAV_CH3, MAV_CH5, WL_CH1, WL_CH3, WL_CH5, Ratio_CH1, Ratio_CH3, Ratio_CH5]`

---

## 三种建模方案概览

| 方案 | 信号处理 | 特征/输入 | 模型 | Test |
|------|---------|----------|------|------|
| 基线 | — | 9 维 | RandomForest | 0.667 |
| 路线A | VMD 分解 | 111 维多域 | ExtraTrees | 0.768 |
| 路线B | CWT scalogram | [3,64,128] | 2D-CNN | 0.747 |

三方共用 `src/preprocess/` 的信号准备，和相同的 train/test 划分（random_state=42）保证可对比。

---

## 信号分解模块（src/decompose/）

### vmd.py — VMD 变分模态分解
- `vmd_decompose(signal_1ch)`：单通道 → K=4 个 IMF（按中心频率低→高排序）
- `vmd_decompose_3ch`：三通道分别分解 → [3, K, T]
- 参数：alpha=2000（带宽约束），K=4，tol=1e-7
- VMD 要求偶数长度，奇数自动截断末点

### cwt.py — CWT scalogram 时频图
- `cwt_scalogram(signal_1ch)`：单通道 → [64尺度, 128时间]，Morlet 小波
- `cwt_scalogram_3ch`：三通道 → [3, 64, 128]
- **关键：全局 log 归一化**（非每通道独立），保留通道间相对能量，CNN 准确率因此 +20%
- 时间轴重采样到固定宽度 128，统一变长片段

### features_rich.py — 111 维多域特征
- `extract_rich_features(segment_3ch)`：VMD 分解后每 IMF 提取 9 特征
- 每 IMF 特征：MAV/RMS/WL/ZCR/KURT/SKEW（时域）+ FC/BW/PF（频域）
- 维度：3通道 × 4 IMF × 9 = 108，加 3 维 Ratio = 111

---

## 路线A（src/routeA/）

- `pipeline_a.py`：读 all_features.csv 元信息 → 对每个 .npy 提 111 维 → rich_features.csv
- `train_a.py`：ExtraTrees（300树，balanced）+ 5折交叉验证；分层划分同 random_state=42
- `evaluate_a.py`：分类报告 + 混淆矩阵 + 各环境准确率 + 特征重要度
- 模型实验对比：ExtraTrees(0.768) > RF(0.758) > HistGB(0.707) > SVM(0.687)

## 路线B（src/routeB/）

- `dataset_b.py`：预计算 scalogram 缓存 + 数据增强（时移/缩放/噪声/频带遮挡）+ train/val/test 划分
- `model_b.py`：轻量 2D-CNN，4 个 Conv 块(32→64→128→128) + GAP + Dropout(0.4)，~26万参数
- `train_b.py`：Adam + 余弦退火 + 标签平滑(0.05) + 早停(patience=30)；weight_decay=3e-4
- `evaluate_b.py`：分类报告 + 混淆矩阵 + 训练曲线
- 防过拟合关键：全局归一化 + 适度增强（过强会欠拟合）+ GAP 替代大 FC

---

## 最终主模型（src/model.py + src/train.py + src/evaluate.py）

### 模型：LightGBM + Top-100 特征选择

经系统实验对比（ExtraTrees/RF/SVM/MLP/XGBoost/LightGBM），LightGBM + Top-100 特征效果最优：

| 方法 | 5种子平均 |
|------|---------|
| ExtraTrees-800 (全232维) | 0.816±0.018 |
| LightGBM-500 (全232维) | 0.825±0.025 |
| **LightGBM-800 (Top-100维)** | **0.847±0.017** |

**关键发现**：Top-100 特征比全 232 维更好——噪声维度会降低准确率。
用 ExtraTrees 重要度选出 Top-100，再喂给 LightGBM。

**LightGBM 参数**：n_estimators=800, num_leaves=63, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, min_child_samples=10, class_weight=balanced

### 数据划分（src/dataset.py）

- 读取 `enhanced_features.csv`，按手势标签分层抽样 80/20
- 不按环境分层（最大化判别效果）
- 划分结果保存到 `final_split.csv`

### 训练（src/train.py）

1. 加载增强特征（`enhanced_features.csv`，若无则自动提取）
2. 按标签分层划分 train/test（random_state=42）
3. ExtraTrees 重要度选 Top-100 特征
4. LightGBM 训练 + 5折CV
5. 保存 `checkpoints/lgbm_model.pkl` 和 `checkpoints/top_feature_idx.pkl`

### 评估（src/evaluate.py）

- 整体 classification_report + 混淆矩阵
- 各环境单独准确率
- LightGBM 特征重要度（Top-20）

## 评估（src/evaluate.py）

- 整体 classification_report + 混淆矩阵
- 各环境单独准确率
- 特征重要度排名（Gini importance）

---

## 原始脚本（存档）

### PDMA.py — 原始信号清洗与特征提取

整体流程已被拆分到 `cleaner.py` + `features.py`。原始版本处理整个文件（不切分），包含滑动窗口空间特征（`calculate_spatial_features`）和统计摘要，这些在当前流程中未使用。

### SNR.py — 原始信噪比分析

激活检测逻辑已被移植到 `segmenter.py`。原始版本包含完整的 SNR 计算和论文配图导出功能，这些在分类流程中不直接使用。

---

## 修改记录

| 日期 | 文件 | 变更内容 |
|------|------|---------|
| 2026-06-05 | — | 初始建档，记录 PDMA.py 和 SNR.py 原始逻辑 |
| 2026-06-05 | 全部 | 项目重构：数据整理到 data/raw/、预处理拆分到 src/preprocess/、创建 MLP 模型代码框架 |
| 2026-06-05 | segmenter.py | 修复 sc 手势切分失败：merge_gap_s 从 0.050→0.200，适配复合手势的多子 burst 合并。总样本 299→493 |
| 2026-06-05 | pipeline.py | 修复 features 目录不存在时的写入报错（添加 mkdir） |
| 2026-06-05 | notebooks/ | 重写两个 ipynb：01 完整预处理文档（6 节含详细说明），02 参数调优案例文档（5 节含对比分析） |
| 2026-06-05 | model/dataset/train/evaluate | MLP→随机森林；数据划分改为按(环境,手势)分层 80/20 |
| 2026-06-05 | notebooks/ | 新增 03_training_results.ipynb：训练结果分析（7 节：划分、混淆矩阵、各环境评估、特征重要度、误分类、参数敏感性） |
| 2026-06-06 | src/decompose/ | 新增信号分解模块：vmd.py(VMD分解)、cwt.py(CWT scalogram)、features_rich.py(111维多域特征) |
| 2026-06-06 | src/routeA/ | 路线A：VMD+111维多域特征+ExtraTrees，test 0.768（基线0.667）。新增 notebook 04 |
| 2026-06-06 | src/routeB/ | 路线B：CWT scalogram+轻量2D-CNN，test 0.747。关键：全局log归一化保留通道间能量(+20%)。新增 notebook 05 |
| 2026-06-06 | requirements/README | 加 PyWavelets/EMD-signal/vmdpy；README 增三方案对比 |
| 2026-06-06 | src/decompose/features_enhanced.py | 新增232维增强特征(Hjorth/熵/Hilbert包络/小波包/通道相关)。单片段5种子真实均值0.796 |
| 2026-06-06 | notebooks/ | 新增 06_summary_comparison.ipynb：多方法5种子真实对比+学习曲线+数据量建议。文件级投票达0.863，外推达90%需每类~69片段(1.8倍数据) |
| 2026-06-06 | data/processed/method_results.json | 缓存所有方法的多种子实验结果+学习曲线外推，供notebook读取 |
| 2026-06-06 | data/raw/base/ | 补充新采集数据（-3后缀，10个文件），base从20→30文件。总片段493→1210，base片段245→962 |
| 2026-06-06 | src/decompose/features_enhanced.py + Stacking | V2数据重跑：增强特征单片段0.816±0.018；Stacking(ET+SVM+RF)0.842±0.021；文件投票0.853±0.020 |
| 2026-06-06 | notebooks/06 | 更新汇总notebook：V1/V2对比、学习曲线（显示饱和趋势）、数据量建议（手势4/sc是唯一瓶颈，补采达90%） |
| 2026-06-11 | src/decompose/features_temporal.py | 新增117维时序特征(时间分段剖面/IMF时序/包络峰值)，专门捕获sc双峰结构 |
| 2026-06-11 | src/model.py | 升级为4模型软投票(2×LGBM+ET+SVM)，5种子均值0.870±0.021 |
| 2026-06-11 | src/train.py | 重写：加载增强+时序特征(349维)→Top-100选择→4模型软投票训练 |
| 2026-06-11 | src/evaluate.py | 更新为集成模型评估 |
