import numpy as np
import os

d = "C:/sd/ddpm-generalization-theory/data-processing/data/hcmc/processed"

for f in ["b_canvas.npy", "g_canvas.npy", "r_canvas.npy"]:
    path = os.path.join(d, f)
    if os.path.exists(path):
        arr = np.load(path)
        print(
            f"{f}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, nonzero={(arr > 0).sum()}, mean={arr.mean():.2f}"
        )
    else:
        print(f"{f}: NOT FOUND")

# Check labels
labels_path = os.path.join(d, "cells_256x256_labels.csv")
if os.path.exists(labels_path):
    import pandas as pd

    df = pd.read_csv(labels_path)
    print(f"\nLabels: {len(df)} rows")
    print(df.columns.tolist())
    print(df.head())
else:
    print("Labels CSV not found")
