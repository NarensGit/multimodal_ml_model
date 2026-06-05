#!/usr/bin/env python3
"""Build a balanced binary engagement dataset from the processed multimodal features.

This converts condition-based proxy labels into a binary target:
  - Baseline condition -> Not Engaged (0)
  - LPE condition      -> Not Engaged (0)
  - HPE condition      -> Engaged (1)

Because the raw dataset does not contain direct human engagement annotations,
the binary target is derived from the experimental condition labels.

Usage:
  python create_binary_dataset.py --in uploads/processed_features.csv --out data/engagement_binary_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_binary_label(condition: str) -> int:
    condition = str(condition).strip().lower()
    if "hpe" in condition:
        return 1
    if "baseline" in condition or "lpe" in condition:
        return 0
    raise ValueError(f"Unknown condition value: {condition}")


def label_name(value: int) -> str:
    return "Engaged" if int(value) == 1 else "Not Engaged"


def build_binary_dataset(input_path: Path, output_path: Path, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "condition" not in df.columns:
        raise ValueError("Input CSV must contain a 'condition' column.")

    df = df.copy()
    df["engaged"] = df["condition"].apply(infer_binary_label).astype(int)
    df["engaged_label"] = df["engaged"].apply(label_name)

    # Keep traceability, but remove the old placeholder label if present.
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Balance classes by undersampling the majority class.
    counts = df["engaged"].value_counts()
    target = int(counts.min())
    parts = []
    for cls_value in sorted(counts.index.tolist()):
        class_df = df[df["engaged"] == cls_value]
        class_df = class_df.sample(n=target, random_state=seed).sort_values(["participant", "start_time"])
        parts.append(class_df)

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Put the binary target near the front for readability.
    front_cols = [c for c in ["participant", "condition", "start_time", "engaged", "engaged_label"] if c in balanced.columns]
    other_cols = [c for c in balanced.columns if c not in front_cols]
    balanced = balanced[front_cols + other_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_path, index=False)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_path",
        default="uploads/processed_features.csv",
        help="Input processed CSV with condition and feature columns",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/engagement_binary_dataset.csv",
        help="Output balanced binary dataset CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    balanced = build_binary_dataset(input_path, output_path, seed=args.seed)

    print(f"Wrote: {output_path}")
    print("Rows:", len(balanced))
    print("Class counts:")
    print(balanced["engaged"].value_counts().sort_index().to_string())
    print("Label names:")
    print(balanced["engaged_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
