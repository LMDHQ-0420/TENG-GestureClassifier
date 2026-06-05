---
name: project-teng-gesture-classifier
description: TENG手势分类项目的基本数据与代码结构
metadata:
  type: project
---

TENG手势识别项目，使用压电信号（TENG传感器）采集手势数据，通道1、3、5（CSV第0、2、4列），采样率1000 Hz。

**数据集结构：**
- 基础/：20次动作/文件，训练用
- 风噪60-85db/：20次动作/文件，抗噪测试
- 基础（可用于验证）/：5次动作/文件
- 辐照40mm-30min/：5次动作/文件
- 手势类别：1, 2, 3, 4, 5, go the way, ok, sc, stop, wave

**代码文件：**
- PDMA.py：信号清洗（带通+陷波+谱减法）+ 特征提取（9维向量、滑动窗口Ratio），输出cleaned/vector/features/summary四类CSV
- SNR.py：信噪比分析，检测联合动作段，导出论文配图专用CSV（A/B/C/D四小图），含纯净度自检和统一X轴对齐

**Why:** 项目目标是验证TENG传感器在不同环境（风噪、辐照）下的手势识别鲁棒性。

**How to apply:** 修改代码后必须同步更新 code_guide.md 的修改记录表。
