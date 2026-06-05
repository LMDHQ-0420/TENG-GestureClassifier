"""数据重命名与整理脚本（一次性运行）

将中文命名的原始数据复制到 data/raw/ 英文目录结构：
- 基础/ + 基础（可用于验证）/ → data/raw/base/（合并，后缀 -1/-2 区分来源）
- 风噪60-85db/ → data/raw/wind_noise/
- 辐照40mm-30min/ → data/raw/uv_radiation/
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据目录
SRC_BASE_TRAIN = PROJECT_ROOT / "基础"
SRC_BASE_VAL = PROJECT_ROOT / "基础（可用于验证）"
SRC_WIND = PROJECT_ROOT / "风噪60-85db"
SRC_UV = PROJECT_ROOT / "辐照40mm-30min"

# 目标目录
DST_BASE = PROJECT_ROOT / "data" / "raw" / "base"
DST_WIND = PROJECT_ROOT / "data" / "raw" / "wind_noise"
DST_UV = PROJECT_ROOT / "data" / "raw" / "uv_radiation"


def sanitize_name(name: str) -> str:
    """文件名中的空格替换为下划线"""
    return name.replace(" ", "_")


def copy_env(src_dir: Path, dst_dir: Path, suffix: str = ""):
    """复制一个环境目录下的所有 CSV 文件到目标目录

    Parameters
    ----------
    src_dir : 源目录
    dst_dir : 目标目录
    suffix : 追加到文件名（无扩展名部分）的后缀，如 "-1"
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for csv_file in sorted(src_dir.glob("*.csv")):
        stem = sanitize_name(csv_file.stem)
        new_name = f"{stem}{suffix}.csv"
        dst_path = dst_dir / new_name
        shutil.copy2(csv_file, dst_path)
        print(f"  {csv_file.name}  →  {dst_path.relative_to(PROJECT_ROOT)}")
        count += 1
    return count


def main():
    print("=" * 60)
    print("数据重命名与整理")
    print("=" * 60)

    total = 0

    print(f"\n[1/4] 基础（训练）→ data/raw/base/ （后缀 -1）")
    total += copy_env(SRC_BASE_TRAIN, DST_BASE, suffix="-1")

    print(f"\n[2/4] 基础（验证）→ data/raw/base/ （后缀 -2）")
    total += copy_env(SRC_BASE_VAL, DST_BASE, suffix="-2")

    print(f"\n[3/4] 风噪60-85db → data/raw/wind_noise/")
    total += copy_env(SRC_WIND, DST_WIND)

    print(f"\n[4/4] 辐照40mm-30min → data/raw/uv_radiation/")
    total += copy_env(SRC_UV, DST_UV)

    print(f"\n完成！共复制 {total} 个文件。")

    # 验证
    for name, d in [("base", DST_BASE), ("wind_noise", DST_WIND), ("uv_radiation", DST_UV)]:
        n = len(list(d.glob("*.csv")))
        print(f"  {name}: {n} files")


if __name__ == "__main__":
    main()
