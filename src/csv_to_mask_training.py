# src/csv_to_mask_training.py (FOR TRAINING - SAVES GRAYSCALE MASKS)
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def csv_to_grayscale_mask(csv_path, image_shape):
    """
    Convert BUU-LSPINE CSV to grayscale segmentation mask for training.
    
    Args:
        csv_path: Path to CSV file
        image_shape: (height, width) of original image
    
    Returns:
        mask: uint8 array with values 0-5 (0=background, 1=L1, 2=L2, 3=L3, 4=L4, 5=L5)
    """
    # Read CSV (no header)
    df = pd.read_csv(csv_path, header=None)
    
    # Create empty mask (all background = 0)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Vertebra order (L1 to L5 only)
    vertebra_names = ['L1', 'L2', 'L3', 'L4', 'L5']
    
    for v_idx, vertebra in enumerate(vertebra_names):
        row_idx = v_idx * 2
        
        # Check if we have enough rows
        if row_idx + 1 >= len(df):
            continue
        
        # Get upper and lower edge rows
        upper = df.iloc[row_idx]
        lower = df.iloc[row_idx + 1]
        
        # Create polygon from 4 corner points
        polygon = np.array([
            [int(upper[0]), int(upper[1])],  # top-left
            [int(upper[2]), int(upper[3])],  # top-right
            [int(lower[2]), int(lower[3])],  # bottom-right
            [int(lower[0]), int(lower[1])]   # bottom-left
        ], dtype=np.int32)
        
        # Fill polygon with class ID (v_idx + 1)
        # Class IDs: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5
        cv2.fillPoly(mask, [polygon], v_idx + 1)
    
    return mask

def batch_convert():
    """Convert all CSV files to grayscale training masks"""
    image_dir = Path("data/raw/images")
    annotation_dir = Path("data/raw/annotations")
    output_dir = Path("data/processed/training_masks")  # Separate folder for training masks
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CONVERTING CSV TO GRAYSCALE TRAINING MASKS")
    print("=" * 60)
    print(f"Image dir: {image_dir.absolute()}")
    print(f"Annotation dir: {annotation_dir.absolute()}")
    print(f"Output dir: {output_dir.absolute()}")
    
    # Check if directories exist
    if not image_dir.exists():
        print(f"ERROR: {image_dir} not found!")
        return
    
    if not annotation_dir.exists():
        print(f"ERROR: {annotation_dir} not found!")
        return
    
    csv_files = list(annotation_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("No CSV files found!")
        return
    
    print("\nTraining mask format:")
    print("  Pixel values: 0 = Background")
    print("  Pixel values: 1 = L1")
    print("  Pixel values: 2 = L2")
    print("  Pixel values: 3 = L3")
    print("  Pixel values: 4 = L4")
    print("  Pixel values: 5 = L5")
    print("  Output format: {filename}_mask.png (grayscale)\n")
    
    success_count = 0
    for csv_path in tqdm(csv_files, desc="Converting"):
        # Find corresponding image to get dimensions
        image_path = image_dir / f"{csv_path.stem}.jpg"
        
        if not image_path.exists():
            print(f"Warning: No image for {csv_path.name}")
            continue
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Cannot load {image_path}")
            continue
        
        # Create grayscale mask (values 0-5)
        mask = csv_to_grayscale_mask(csv_path, image.shape)
        
        # Save as grayscale PNG
        output_path = output_dir / f"{csv_path.stem}_mask.png"
        cv2.imwrite(str(output_path), mask)
        success_count += 1
    
    print(f"\n✓ Saved {success_count} training masks to {output_dir}")
    
    # Verify
    verify_masks(output_dir)

def verify_masks(mask_dir):
    """Verify masks have correct class IDs (0-5)"""
    mask_files = list(mask_dir.glob("*_mask.png"))[:3]
    
    print("\n" + "=" * 60)
    print("VERIFYING TRAINING MASKS")
    print("=" * 60)
    
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique = np.unique(mask)
        print(f"\n{mask_path.name}")
        print(f"  Shape: {mask.shape}")
        print(f"  Unique values: {unique}")
        print(f"  Min: {mask.min()}, Max: {mask.max()}")
        
        if set(unique) == {0, 1, 2, 3, 4, 5}:
            print(f"  ✅ PERFECT! Ready for U-Net/Mask R-CNN training")
        elif set(unique).issubset({0, 1, 2, 3, 4, 5}):
            print(f"  ⚠️ Missing some classes, but acceptable")
        else:
            print(f"  ❌ ERROR: Unexpected values! Should be 0-5 only")

if __name__ == "__main__":
    batch_convert()