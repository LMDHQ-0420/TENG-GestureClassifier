"""批量运行预处理流水线

处理 data/raw/ 下三个环境的所有 CSV 文件，输出到 data/processed/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.pipeline import run_all

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("TENG 手势信号预处理流水线")
    print("=" * 60)

    all_features = run_all(RAW_ROOT, PROCESSED_ROOT)

    print(f"\n总计：{len(all_features)} 个有效子动作片段")
    print(f"特征文件：{PROCESSED_ROOT / 'features' / 'all_features.csv'}")
    print(f"统计文件：{PROCESSED_ROOT / 'stats' / 'segment_counts.csv'}")


if __name__ == "__main__":
    main()
