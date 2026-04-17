# verify_classes.py
import cv2
import numpy as np
from pathlib import Path

# Check a mask from your standardized folder
mask_path = Path("data/standardized/masks/0001-F-037Y1_mask.png")

if mask_path.exists():
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    unique = np.unique(mask)
    
    print("=" * 40)
    print("MULTICLASS LABEL VERIFICATION")
    print("=" * 40)
    print(f"File: {mask_path.name}")
    print(f"Unique pixel values: {unique}")
    print(f"Max value: {mask.max()}")
    print()
    
    if set(unique) == {0, 1, 2, 3, 4, 5}:
        print("✅ PERFECT! You have:")
        print("   0 = Background")
        print("   1 = L1")
        print("   2 = L2")
        print("   3 = L3")
        print("   4 = L4")
        print("   5 = L5")
    elif mask.max() == 5:
        print("✅ Correct! Classes 1-5 (L1-L5) present")
        print("   (S1/class 6 not in this dataset)")
    else:
        print("❌ Something wrong")
else:
    print("Run csv_to_mask_training.py first")