"""
00_prepare_dataset.py

Purpose:
    Freeze the dataset used for the ANN Challenge so that every later
    script (main challenge model + bonus challenge model) reads the
    exact same data. This satisfies the "Do not change the dataset" rule.

Dataset:
    Breast Cancer Wisconsin (Diagnostic) Dataset
    - 569 samples, 30 numeric features, 1 binary target
      (0 = malignant, 1 = benign)
    - Loaded from scikit-learn's built-in datasets, so it is a public,
      well known, ready-made dataset (no scraping, no manual labeling).

Output:
    outputs/dataset.csv        -> the frozen dataset
    outputs/dataset_summary.txt -> a short text summary of the data
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT_CSV = "/home/claude/ann_project/outputs/dataset.csv"
OUT_SUMMARY = "/home/claude/ann_project/outputs/dataset_summary.txt"

def main():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(OUT_CSV, index=False)

    with open(OUT_SUMMARY, "w") as f:
        f.write("Breast Cancer Wisconsin (Diagnostic) Dataset\n")
        f.write("=" * 45 + "\n")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns (features + target): {df.shape[1]}\n")
        f.write(f"Target classes: {sorted(df['target'].unique().tolist())}\n")
        f.write(f"Class 0 (malignant) count: {(df['target']==0).sum()}\n")
        f.write(f"Class 1 (benign) count: {(df['target']==1).sum()}\n")
        f.write("\nFeature list:\n")
        for c in df.columns[:-1]:
            f.write(f"  - {c}\n")

    print("Dataset saved to", OUT_CSV)
    print(df["target"].value_counts())

if __name__ == "__main__":
    main()
