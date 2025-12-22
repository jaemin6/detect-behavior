import pandas as pd
from pathlib import Path

# results.csv 경로 (src 기준)
RESULTS_PATH = Path(__file__).resolve().parents[2] / "runs/detect/train/results.csv"

df = pd.read_csv(RESULTS_PATH)

before = df[df['epoch'] == 0].iloc[0]
after = df.iloc[-1]

print("=== Before Training (epoch 0) ===")
print(f"mAP50      : {before['metrics/mAP50']:.4f}")
print(f"Recall     : {before['metrics/recall(B)']:.4f}")
print(f"Precision  : {before['metrics/precision(B)']:.4f}")

print("\n=== After Training (last epoch) ===")
print(f"mAP50      : {after['metrics/mAP50']:.4f}")
print(f"Recall     : {after['metrics/recall(B)']:.4f}")
print(f"Precision  : {after['metrics/precision(B)']:.4f}")
