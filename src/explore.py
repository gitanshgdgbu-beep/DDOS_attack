import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def explore_dataset(data_path: str = "data/ddos.csv", output_chart: str = "models/class_dist.png"):
    # Resolve paths relative to project root if needed
    data_file = Path(data_path)
    if not data_file.exists():
        # Try finding relative to current file's parent directory
        alt_path = Path(__file__).resolve().parent.parent / data_path
        if alt_path.exists():
            data_file = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at {data_path} or {alt_path}")

    output_file = Path(output_chart)
    if not output_file.is_absolute() and not output_file.parent.exists():
        alt_out_parent = Path(__file__).resolve().parent.parent / output_file.parent
        if alt_out_parent.exists():
            output_file = alt_out_parent / output_file.name

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_file}...")
    df = pd.read_csv(data_file)

    # 1. Strip column names
    df.columns = df.columns.str.strip()
    print("Column names stripped successfully.\n")

    # 2. Print shape
    print("=" * 50)
    print("DATASET SHAPE")
    print("=" * 50)
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}\n")

    # 3. Print label counts
    print("=" * 50)
    print("LABEL DISTRIBUTION")
    print("=" * 50)
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts()
        label_pcts = df["Label"].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({
            "Count": label_counts,
            "Percentage (%)": label_pcts.round(2)
        })
        print(dist_df)
    else:
        print("Warning: 'Label' column not found in dataset!")
    print()

    # 4. Print missing values
    print("=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"Total missing values across all columns: {total_nulls:,}")
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        print("\nColumns with missing values:")
        for col, count in cols_with_nulls.items():
            print(f"  - {col}: {count:,} ({count / len(df) * 100:.3f}%)")
    else:
        print("No missing values found.")
    print()

    # 5. Print infinite values
    print("=" * 50)
    print("INFINITE VALUES")
    print("=" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    total_infs = inf_counts.sum()
    print(f"Total infinite values across all numeric columns: {total_infs:,}")
    cols_with_infs = inf_counts[inf_counts > 0]
    if not cols_with_infs.empty:
        print("\nColumns with infinite values:")
        for col, count in cols_with_infs.items():
            print(f"  - {col}: {count:,} ({count / len(df) * 100:.3f}%)")
    else:
        print("No infinite values found.")
    print()

    # 6. Save class-distribution bar chart
    if "Label" in df.columns:
        print("=" * 50)
        print("SAVING CLASS DISTRIBUTION CHART")
        print("=" * 50)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        colors = ["#4338CA", "#EF4444", "#10B981", "#F59E0B"]
        bars = ax.bar(
            label_counts.index.astype(str),
            label_counts.values,
            color=colors[:len(label_counts)],
            edgecolor="#1E293B",
            linewidth=1.2,
            width=0.5,
            alpha=0.9
        )

        total_samples = len(df)
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_samples) * 100
            ax.annotate(
                f"{height:,}\n({percentage:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#1E293B"
            )

        ax.set_title("DDoS Dataset Class Distribution", fontsize=15, fontweight="bold", pad=20, color="#0F172A")
        ax.set_xlabel("Class Label", fontsize=12, fontweight="semibold", labelpad=10, color="#1E293B")
        ax.set_ylabel("Sample Count", fontsize=12, fontweight="semibold", labelpad=10, color="#1E293B")
        ax.set_ylim(0, max(label_counts.values) * 1.15)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Class distribution chart saved to: {output_file}\n")


if __name__ == "__main__":
    explore_dataset()
