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

## 模型（src/model.py）

随机森林分类器（scikit-learn `RandomForestClassifier`）：
- 200 棵决策树，`max_features="sqrt"`，`class_weight="balanced"`
- 输入：9D 标准化特征向量
- 输出：10 类手势

## 数据划分（src/dataset.py）

- 读取 `all_features.csv`，按 `(环境, 手势)` 组合分层抽样
- 每组 ~20% 作为测试集，至少保证 1 个测试样本
- **每个环境在 train 和 test 中均有样本**
- 使用 `StandardScaler` 标准化（基于训练集 fit）
- 划分结果保存到 `all_features_split.csv`（含 `split` 列）

## 训练（src/train.py）

- 加载数据 → 分层划分 → 训练随机森林
- 保存模型到 `checkpoints/random_forest.pkl`
- 保存 scaler 到 `checkpoints/scaler.pkl`

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
