# check_classes.py
import cv2
import numpy as np
from pathlib import Path

mask_dir = Path("data/processed/training_masks")
mask_files = list(mask_dir.glob("*_mask.png"))[:3]

print("Checking class IDs in your masks:")
print("=" * 40)

for mask_path in mask_files:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    unique = np.unique(mask)
    print(f"\n{mask_path.name}:")
    print(f"  Unique values: {unique}")
    print(f"  Max class ID: {mask.max()}")

print("\n" + "=" * 40)
print("If max = 5, you have L1-L5 only (no S1)")
print("If max = 6, you have S1 as well")