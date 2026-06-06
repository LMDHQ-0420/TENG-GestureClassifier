"""VMD（变分模态分解）封装

VMD 将信号分解为 K 个有限带宽的本征模态函数（IMF），各模态围绕各自的中心频率紧凑分布。
相比 EMD，VMD 数学上更严谨、无模态混叠、对噪声更鲁棒，分解结果适合论文配图。
"""

from typing import Tuple

import numpy as np
from vmdpy import VMD

# VMD 参数
VMD_PARAMS = {
    "alpha": 2000,   # 带宽约束（越大模态越紧凑）
    "tau": 0,        # 噪声容忍（0 表示严格重构）
    "K": 4,          # 分解的模态数量
    "DC": 0,         # 不强制第一个模态为直流
    "init": 1,       # 中心频率均匀初始化
    "tol": 1e-7,     # 收敛容差
}


def vmd_decompose(signal_1ch: np.ndarray, params: dict = None) -> np.ndarray:
    """对单通道信号做 VMD 分解

    Parameters
    ----------
    signal_1ch : shape [T]
    params : VMD 参数，None 使用默认

    Returns
    -------
    imfs : shape [K, T']，K 个 IMF 模态（T' 可能比 T 少 1，VMD 要求偶数长度）
        模态按中心频率从低到高排列
    """
    if params is None:
        params = VMD_PARAMS

    # VMD 要求信号长度为偶数，奇数则截断最后一个点
    sig = np.asarray(signal_1ch, dtype=float)
    if len(sig) % 2 != 0:
        sig = sig[:-1]

    u, u_hat, omega = VMD(
        sig,
        params["alpha"], params["tau"], params["K"],
        params["DC"], params["init"], params["tol"],
    )
    # u shape: [K, T]，按最终中心频率排序确保低频→高频
    final_omega = omega[-1, :]
    order = np.argsort(final_omega)
    return u[order]


def vmd_decompose_3ch(signal_3ch: np.ndarray, params: dict = None) -> np.ndarray:
    """对三通道信号分别做 VMD 分解

    Parameters
    ----------
    signal_3ch : shape [T, 3]

    Returns
    -------
    imfs : shape [3, K, T']，每个通道 K 个 IMF
    """
    if params is None:
        params = VMD_PARAMS
    channel_imfs = [vmd_decompose(signal_3ch[:, ch], params) for ch in range(signal_3ch.shape[1])]
    return np.stack(channel_imfs, axis=0)
