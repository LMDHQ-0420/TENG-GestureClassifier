"""统一 CSV 读取与手势标签映射

来源：SNR.py read_csv_smart 的多编码探测逻辑，简化为本项目专用版本。
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# 通道索引（CSV 第 0/2/4 列 → CH1/CH3/CH5）
CHANNEL_COLS = [0, 2, 4]

# 手势名 → 整数标签（按字母顺序排列）
GESTURE_NAMES: List[str] = [
    "1", "2", "3", "4", "5",
    "go_the_way", "ok", "sc", "stop", "wave",
]
_GESTURE_TO_LABEL = {name: idx for idx, name in enumerate(GESTURE_NAMES)}


def load_raw_csv(path: Path, channel_cols: list = None) -> np.ndarray:
    """读取原始无表头 CSV，返回指定通道数据

    Parameters
    ----------
    path : CSV 文件路径
    channel_cols : 要提取的列索引，默认 [0, 2, 4]

    Returns
    -------
    np.ndarray, shape [N, len(channel_cols)]
    """
    if channel_cols is None:
        channel_cols = CHANNEL_COLS

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, header=None, encoding=enc)
            return df.iloc[:, channel_cols].values.astype(np.float64)
        except Exception:
            continue
    raise RuntimeError(f"无法读取文件：{path}")


def get_gesture_label(filename: str) -> int:
    """从文件名推断手势标签

    支持格式：'sc-1.csv' → 'sc' → 7，'go_the_way.csv' → 'go_the_way' → 5
    """
    stem = Path(filename).stem
    # 去掉来源后缀（-1, -2）
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
        gesture_name = parts[0]
    else:
        gesture_name = stem

    if gesture_name not in _GESTURE_TO_LABEL:
        raise ValueError(f"未知手势名: '{gesture_name}'（来自文件 '{filename}'）")
    return _GESTURE_TO_LABEL[gesture_name]


def get_gesture_name(filename: str) -> str:
    """从文件名提取手势名（不含来源后缀）"""
    stem = Path(filename).stem
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
        return parts[0]
    return stem
