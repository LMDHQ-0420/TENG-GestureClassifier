# TENG 手势识别系统

基于摩擦纳米发电机（TENG）传感器的手势分类系统。传感器采集手部 3 通道电压信号（CH1/CH3/CH5，采样率 1000 Hz），在三种环境下识别 10 种手势。

## 数据集

| 环境 | 片段数 | 测试集 |
|------|--------|--------|
| 正常环境 (base) | 934 | 20% 分层采样 |
| 风噪环境 (wind_noise) | 174 | 20% 分层采样 |
| 紫外辐照 (uv_radiation) | 67 | 每类 1 个样本 |

手势类别（10类）：`1` `2` `3` `4` `5` `go_the_way` `ok` `sc` `stop` `wave`

## 结果

| 场景 | 测试样本 | 准确率 |
|------|----------|--------|
| 正常环境 | 187 | 90.9% |
| 风噪环境 | 35 | 91.4% |
| 紫外辐照 | 10 | 80.0% |
| **总体** | **232** | **90.5%** |

## 方案

**特征**：VMD（K=4）各 IMF 的 15 维统计特征（232D）+ 时序剖面特征（117D）+ 包络特征（24D）= 373D，经 log 变换后 ExtraTrees Top-100 选择。

**模型**：Transformer 融合模型（1D-Conv stem + TransformerEncoder，0.43M 参数）与 4 模型 sklearn 集成（2×LightGBM + ExtraTrees + SVC）加权融合。uv_radiation 场景使用 TTA×15 + ×8 过采样补偿数据量不足。

## 项目结构

```
├── data/processed/
│   ├── segments/          单动作片段 (.npy)
│   └── features/          特征文件 (.csv / .npy)
├── src/
│   ├── preprocess/        信号预处理（切分、滤波）
│   ├── decompose/         VMD、小波包、特征提取
│   ├── train_transformer.py   训练入口
│   └── model.py           特征选择、sklearn 集成
├── scripts/
│   └── save_predictions.py    重新生成推断结果 npy
├── checkpoints/           模型权重与推断结果
├── notebooks/
│   └── results_visualization.ipynb   结果可视化
└── svg/                   输出图表
```

## 环境配置

```bash
conda create -n TENG-GestureClassifier python=3.11
conda activate TENG-GestureClassifier
pip install -r requirements.txt
# PyTorch 根据平台单独安装，参考 https://pytorch.org/get-started/locally/
```

## 运行

```bash
# 训练
python -m src.train_transformer

# 更新推断结果（训练后执行）
python scripts/save_predictions.py
```

训练完成后，打开 `notebooks/results_visualization.ipynb` 查看结果。
