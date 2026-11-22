import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def plot_and_save_feature_importance(model, feature_names, model_name, output_dir, top_n=20):
    """Generates and saves feature importance plot and CSV to output_dir."""
    output_dir = Path(output_dir)

    # Extract feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier"), "coef_"):
        importances = np.abs(model.named_steps["classifier"].coef_[0])
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Model {model_name} does not have feature importances or coefficients.")
        return

    # Create DataFrame and sort
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Save complete ranking to CSV
    csv_path = output_dir / "feature_importance.csv"
    feat_df.to_csv(csv_path, index=False)
    print(f"Saved full feature importance rankings to: {csv_path}")

    # Plot top N features
    top_df = feat_df.head(top_n).sort_values(by="Importance", ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Use a gradient bar color
    norm = plt.Normalize(top_df["Importance"].min(), top_df["Importance"].max())
    colors = plt.cm.viridis(norm(top_df["Importance"].values))

    bars = ax.barh(
        top_df["Feature"],
        top_df["Importance"],
        color=colors,
        edgecolor="#1E293B",
        linewidth=0.8,
        height=0.65
    )

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="semibold",
            color="#1E293B"
        )

    ax.set_title(f"Top {top_n} Feature Importances ({model_name})", fontsize=14, fontweight="bold", pad=15, color="#0F172A")
    ax.set_xlabel("Relative Importance Score", fontsize=11, fontweight="semibold", labelpad=10, color="#1E293B")
    ax.set_ylabel("Network Traffic Feature", fontsize=11, fontweight="semibold", labelpad=10, color="#1E293B")
    ax.set_xlim(0, max(top_df["Importance"]) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to: {plot_path}")


def plot_and_save_confusion_matrices(results, output_dir):
    """Generates and saves multi-model confusion matrix heatmaps to output_dir."""
    output_dir = Path(output_dir)
    n_models = len(results)
    class_names = ["BENIGN (0)", "DDoS (1)"]

    plt.style.use("default")
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=300)
    if n_models == 1:
        axes = [axes]

    cmaps = ["Blues", "Purples", "Greens"]

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        total = np.sum(cm)

        # Annotations showing count and percentage
        annot_matrix = np.array([
            [f"{cm[0, 0]:,}\n({cm[0, 0] / total * 100:.1f}%)\n[TN]", f"{cm[0, 1]:,}\n({cm[0, 1] / total * 100:.1f}%)\n[FP]"],
            [f"{cm[1, 0]:,}\n({cm[1, 0] / total * 100:.1f}%)\n[FN]", f"{cm[1, 1]:,}\n({cm[1, 1] / total * 100:.1f}%)\n[TP]"]
        ])

        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap=cmaps[idx % len(cmaps)],
            cbar=False,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"fontsize": 11, "fontweight": "semibold"},
            linewidths=1.5,
            linecolor="white"
        )

        acc_pct = res["accuracy"] * 100
        f1_pct = res["f1"] * 100
        ax.set_title(f"{name}\nAcc: {acc_pct:.2f}% | F1: {f1_pct:.2f}%", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=8, color="#1E293B")
        ax.set_ylabel("True Class", fontsize=11, fontweight="semibold", labelpad=8, color="#1E293B")

    plt.suptitle("Model Confusion Matrices Comparison", fontsize=16, fontweight="bold", y=1.03, color="#0F172A")
    plt.tight_layout()

    out_file = output_dir / "confusion_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {out_file}")

    # Also save as confusion_matrices.png for convenience
    alt_out_file = output_dir / "confusion_matrices.png"
    import shutil
    shutil.copyfile(out_file, alt_out_file)


def save_results_comparison_table(results, best_model_name, output_dir):
    """Generates both CSV and visual table graphic of model comparisons."""
    output_dir = Path(output_dir)

    # Build DataFrame
    rows = []
    for m_name, res in results.items():
        is_best = " (Best)" if m_name == best_model_name else ""
        rows.append({
            "Model": f"{m_name}{is_best}",
            "Accuracy": f"{res['accuracy'] * 100:.2f}%",
            "Precision": f"{res['precision'] * 100:.2f}%",
            "Recall": f"{res['recall'] * 100:.2f}%",
            "F1-Score": f"{res['f1'] * 100:.2f}%"
        })

    summary_df = pd.DataFrame(rows)

    # 1. Save CSV
    csv_file = output_dir / "model_comparison.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"Saved results comparison CSV to: {csv_file}")

    # 2. Save styled high-resolution table image
    fig, ax = plt.subplots(figsize=(9, 2.5), dpi=300)
    ax.axis("off")

    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    table_data = [[row["Model"], row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"]] for row in rows]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header and rows
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(1.0)
        if row_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(11)
        else:
            # Highlight best model row with subtle soft green tint
            if best_model_name in rows[row_idx - 1]["Model"]:
                cell.set_facecolor("#ECFDF5")
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#065F46")
            else:
                cell.set_facecolor("#F8FAFC" if row_idx % 2 == 1 else "white")
                cell.get_text().set_color("#1E293B")

    plt.title("Model Performance Benchmark Summary", fontsize=14, fontweight="bold", pad=15, color="#0F172A")
    plt.tight_layout()

    img_file = output_dir / "model_comparison.png"
    plt.savefig(img_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved results comparison table image to: {img_file}")


def main():
    # ---------------------------------------------------------
    # 1. Resolve file paths
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "ddos.csv"
    model_dir = project_root / "models"
    model_save_path = model_dir / "ddos_model.pkl"

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from: {data_path}")
    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # 2. Clean the data
    # ---------------------------------------------------------
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()
    print(f"Initial dataset shape: {df.shape}")

    # Replace infinite values with NaN
    print("Replacing infinite values with NaN...")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop missing values (NaN)
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df):,} rows containing NaN.")

    # Drop duplicate records
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before_dedup - len(df):,} duplicate rows.")
    print(f"Cleaned dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 3. Sample 100k rows if dataset is larger
    # ---------------------------------------------------------
    if len(df) > 100000:
        print(f"Dataset has {len(df):,} rows (> 100,000). Sampling 100,000 rows...")
        df = df.sample(n=100000, random_state=42)
    else:
        print(f"Dataset size is {len(df):,} rows (<= 100,000). Using all rows.")

    # ---------------------------------------------------------
    # 4. Convert Label to binary: BENIGN = 0, else 1
    # ---------------------------------------------------------
    print("Encoding target label (BENIGN=0, DDoS/Other=1)...")
    df["Label"] = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    print("Label distribution in sampled data:")
    print(df["Label"].value_counts().rename(index={0: "BENIGN (0)", 1: "DDoS (1)"}))

    # ---------------------------------------------------------
    # 5. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=["Label"])
    y = df["Label"]
    feature_names = list(X.columns)
    print(f"Number of input features: {len(feature_names)}")

    # ---------------------------------------------------------
    # 6. Stratified 80/20 train-test split
    # ---------------------------------------------------------
    print("Splitting data into 80% train and 20% test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]:,}, Testing samples: {X_test.shape[0]:,}\n")

    # ---------------------------------------------------------
    # 7. Define models to train
    # ---------------------------------------------------------
    # Logistic Regression is wrapped in a Pipeline with StandardScaler
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    # ---------------------------------------------------------
    # 8. Train each model, evaluate, and print metrics
    # ---------------------------------------------------------
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        print("=" * 60)
        print(f"TRAINING: {name}")
        print("=" * 60)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
        }

        print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
        print(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
        print("\nConfusion Matrix:")
        print(f"  [TN: {cm[0, 0]:>6}  |  FP: {cm[0, 1]:>6}]")
        print(f"  [FN: {cm[1, 0]:>6}  |  TP: {cm[1, 1]:>6}]")
        print()

        # Track the best model based on F1-Score (and Accuracy as secondary)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # ---------------------------------------------------------
    # 9. Comparison Summary & Save Best Model
    # ---------------------------------------------------------
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            "Model": m_name,
            "Accuracy": f"{m_res['accuracy'] * 100:.2f}%",
            "Precision": f"{m_res['precision'] * 100:.2f}%",
            "Recall": f"{m_res['recall'] * 100:.2f}%",
            "F1-Score": f"{m_res['f1'] * 100:.2f}%",
        }
        for m_name, m_res in results.items()
    ])
    print(summary_df.to_string(index=False))

    print(f"\nBest Model Selected: {best_model_name} (F1-Score: {best_f1:.4f})")

    # Save best model and feature list together
    payload = {
        "model": best_model_obj,
        "features": feature_names,
        "model_name": best_model_name,
        "metrics": {
            "accuracy": results[best_model_name]["accuracy"],
            "precision": results[best_model_name]["precision"],
            "recall": results[best_model_name]["recall"],
            "f1": results[best_model_name]["f1"],
        },
    }

    joblib.dump(payload, model_save_path)
    print(f"Saved best model and feature list to: {model_save_path}\n")

    # ---------------------------------------------------------
    # 10. Generate and Save Plots & Comparison Table
    # ---------------------------------------------------------
    print("=" * 60)
    print("GENERATING VISUALIZATIONS & ARTIFACTS")
    print("=" * 60)

    # 1. Feature Importance Plot
    plot_and_save_feature_importance(
        model=best_model_obj,
        feature_names=feature_names,
        model_name=best_model_name,
        output_dir=model_dir,
        top_n=20
    )

    # 2. Confusion Matrices Plot
    plot_and_save_confusion_matrices(
        results=results,
        output_dir=model_dir
    )

    # 3. Model Results Comparison Table (CSV + Image)
    save_results_comparison_table(
        results=results,
        best_model_name=best_model_name,
        output_dir=model_dir
    )

    print("\nAll model artifacts and visualizations generated successfully in models/")


if __name__ == "__main__":
    main()
