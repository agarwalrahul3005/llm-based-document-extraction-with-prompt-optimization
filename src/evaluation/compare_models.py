import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Parse filename
# ==========================================================

def parse_filename(filename):

    name = filename.replace(".json", "")

    tokens = name.split("_")

    model = tokens[0]

    ocr = ""

    variant = "Baseline"

    if "easyocr" in tokens:
        ocr = "EasyOCR"

    elif "tesseract" in tokens:
        ocr = "Tesseract"

    if "gepa" in tokens:

        index = tokens.index("evals")

        variant = f"GEPA-{tokens[index + 1]}"

    return model, ocr, variant


# ==========================================================
# Load all results
# ==========================================================

rows = []

for file in sorted(RESULTS_DIR.glob("*.json")):

    model, ocr, variant = parse_filename(file.name)

    with open(file, encoding="utf8") as f:
        metrics = json.load(f)

    rows.append({
        "Model": model,
        "OCR": ocr,
        "Variant": variant,
        "Ground Truth": metrics["ground_truth"],
        "Predictions": metrics["predictions"],
        "Matched": metrics["matched"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["Model", "Variant", "OCR"]
)

print(df)

csv_file = FIGURES_DIR / "overall_results.csv"

df.to_csv(csv_file, index=False)

print(f"\nSaved {csv_file}")


# ==========================================================
# Figure 1
# OCR Comparison
# ==========================================================

baseline = df[df["Variant"] == "Baseline"]

pivot = baseline.pivot(
    index="Model",
    columns="OCR",
    values="F1"
)

ax = pivot.plot(
    kind="bar",
    figsize=(8,5)
)

ax.set_title("EasyOCR vs Tesseract")
ax.set_ylabel("F1 Score")
ax.set_xlabel("Model")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ocr_comparison.png",
    dpi=300
)

plt.close()


# ==========================================================
# Figure 2
# Baseline Models
# ==========================================================

easy = baseline[
    baseline["OCR"] == "EasyOCR"
]

easy = easy.sort_values("F1")

plt.figure(figsize=(8,5))

plt.bar(
    easy["Model"],
    easy["F1"]
)

plt.title("Baseline Model Comparison (EasyOCR)")
plt.ylabel("F1 Score")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "baseline_models.png",
    dpi=300
)

plt.close()


# ==========================================================
# Figure 3
# Precision Recall
# ==========================================================

labels = (
    df["Model"]
    + "\n"
    + df["Variant"]
    + "\n"
    + df["OCR"]
)

x = range(len(df))

width = 0.35

plt.figure(figsize=(14,6))

plt.bar(
    [i-width/2 for i in x],
    df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    [i+width/2 for i in x],
    df["Recall"],
    width,
    label="Recall"
)

plt.xticks(
    x,
    labels,
    rotation=30,
    ha="right"
)

plt.ylabel("Score")

plt.title("Precision vs Recall")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "precision_recall.png",
    dpi=300
)

plt.close()


# ==========================================================
# Figure 4
# GEPA Comparison
# ==========================================================

gepa = df[df["Variant"].str.contains("GEPA")]

if not gepa.empty:

    gepa = gepa.sort_values("Variant")

    labels = (
        gepa["Model"]
        + " ("
        + gepa["OCR"]
        + ")"
        + "\n"
        + gepa["Variant"]
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(labels, gepa["F1"])

    plt.title("GEPA Optimization")
    plt.ylabel("F1 Score")
    plt.ylim(0, max(gepa["F1"]) + 0.05)

    # Show values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "gepa_comparison.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Figure 5
# Overall F1 Comparison
# ==========================================================

plot_df = df.sort_values("F1")

labels = (
    plot_df["Model"]
    + " | "
    + plot_df["OCR"]
    + " | "
    + plot_df["Variant"]
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    labels,
    plot_df["F1"]
)

plt.xlabel("F1 Score")
plt.title("Overall F1 Score Comparison")

plt.xlim(0, max(plot_df["F1"]) + 0.05)

for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 0.003,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "overall_f1_comparison.png",
    dpi=300
)

plt.close()


# ==========================================================
# Summary
# ==========================================================
print("\nGenerated figures:\n")

for file in sorted(FIGURES_DIR.glob("*")):
    print(file.name)
    