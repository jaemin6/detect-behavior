import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================
# Load results.csv
# =====================
RESULTS_PATH = Path(__file__).resolve().parents[2] / "runs/detect/train/results.csv"
df = pd.read_csv(RESULTS_PATH)

# =====================
# Helper function
# =====================
def find_col(keyword):
    for col in df.columns:
        if keyword in col:
            return col
    raise ValueError(f"Column with keyword '{keyword}' not found")

# =====================
# Select columns
# =====================
col_map50 = find_col("mAP50")
col_recall = find_col("recall")
col_precision = find_col("precision")

# =====================
# Before / After
# =====================
before = df.iloc[0]
after = df.iloc[-1]

print("=== Before Training (first epoch) ===")
print(f"Epoch      : {before['epoch']}")
print(f"mAP50      : {before[col_map50]:.4f}")
print(f"Recall     : {before[col_recall]:.4f}")
print(f"Precision  : {before[col_precision]:.4f}")

print("\n=== After Training (last epoch) ===")
print(f"Epoch      : {after['epoch']}")
print(f"mAP50      : {after[col_map50]:.4f}")
print(f"Recall     : {after[col_recall]:.4f}")
print(f"Precision  : {after[col_precision]:.4f}")

# =====================
# Plot 1: Epoch-wise performance
# =====================
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df[col_map50], label='mAP50')
plt.plot(df['epoch'], df[col_recall], label='Recall')
plt.plot(df['epoch'], df[col_precision], label='Precision')

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training Performance Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================
# Plot 2: Before vs After
# =====================
labels = ['mAP50', 'Recall', 'Precision']
before_values = [
    before[col_map50],
    before[col_recall],
    before[col_precision]
]
after_values = [
    after[col_map50],
    after[col_recall],
    after[col_precision]
]

x = range(len(labels))

plt.figure(figsize=(7, 4))
plt.bar(x, before_values, width=0.35, label='Before')
plt.bar([i + 0.35 for i in x], after_values, width=0.35, label='After')

plt.xticks([i + 0.175 for i in x], labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Before vs After Training Performance")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
