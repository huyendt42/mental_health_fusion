from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import config
from scripts.features.orchestrator import FeatureOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract configured feature components.")
    parser.add_argument("--input", default=str(config.TRAIN_PATH), help="Input CSV path.")
    parser.add_argument("--text-col", default=config.TEXT_COL)
    parser.add_argument("--post-id-col", default="post_id")
    parser.add_argument(
        "--split",
        default=None,
        help="Split name used in output filenames. Defaults to the input CSV stem.",
    )
    parser.add_argument(
        "--components",
        default=None,
        help=(
            "Group or sub-extractor selection, e.g. 'affective', "
            "'affective.vader', or 'affective.vad,affective.vader'."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.post_id_col not in df.columns:
        df = df.copy()
        split_name = args.split or Path(args.input).stem
        df[args.post_id_col] = [f"{split_name}_{i}" for i in range(len(df))]
    else:
        split_name = args.split or Path(args.input).stem

    FeatureOrchestrator().extract_dataset(
        df,
        text_col=args.text_col,
        post_id_col=args.post_id_col,
        components=args.components,
        force=args.force,
        split=split_name,
    )


if __name__ == "__main__":
    main()
