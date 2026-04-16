# src/create_splits.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def create_splits():
    """Create train/val/test splits for training"""
    
    # Paths
    original_image_dir = Path("data/raw/images")
    mask_dir = Path("data/processed/training_masks")  # Your grayscale masks
    
    print("=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    print(f"Original images: {original_image_dir.absolute()}")
    print(f"Mask directory: {mask_dir.absolute()}")
    
    # Check if directories exist
    if not original_image_dir.exists():
        print(f"\n❌ ERROR: {original_image_dir} not found!")
        print("   Run csv_to_mask_training.py first.")
        return
    
    if not mask_dir.exists():
        print(f"\n❌ ERROR: {mask_dir} not found!")
        print("   Run csv_to_mask_training.py first.")
        return
    
    # Get all mask files
    mask_files = list(mask_dir.glob("*_mask.png"))
    image_ids = [f.stem.replace("_mask", "") for f in mask_files]
    
    print(f"\nTotal images found: {len(image_ids)}")
    
    if len(image_ids) == 0:
        print("No masks found! Run csv_to_mask_training.py first.")
        return
    
    # Split: train (70%), val (15%), test (15%)
    train_val, test = train_test_split(image_ids, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    
    print(f"\n📊 SPLIT RESULTS:")
    print(f"   Train: {len(train)} images ({len(train)/len(image_ids)*100:.1f}%)")
    print(f"   Val:   {len(val)} images ({len(val)/len(image_ids)*100:.1f}%)")
    print(f"   Test:  {len(test)} images ({len(test)/len(image_ids)*100:.1f}%)")
    
    # Save split lists as text files
    split_dir = Path("data/processed/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    with open(split_dir / "train.txt", "w") as f:
        f.write("\n".join(train))
    with open(split_dir / "val.txt", "w") as f:
        f.write("\n".join(val))
    with open(split_dir / "test.txt", "w") as f:
        f.write("\n".join(test))
    
    print(f"\n✓ Split lists saved to {split_dir}")
    
    # Create organized folder structure
    organized_dir = Path("data/processed/organized")
    
    print(f"\n📁 Creating organized folders in {organized_dir}...")
    
    for split_name, split_ids in [('train', train), ('val', val), ('test', test)]:
        split_img_dir = organized_dir / split_name / "images"
        split_mask_dir = organized_dir / split_name / "masks"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in split_ids:
            # Copy original image
            src_img = original_image_dir / f"{img_id}.jpg"
            dst_img = split_img_dir / f"{img_id}.jpg"
            if src_img.exists():
                shutil.copy(src_img, dst_img)
            
            # Copy mask
            src_mask = mask_dir / f"{img_id}_mask.png"
            dst_mask = split_mask_dir / f"{img_id}_mask.png"
            if src_mask.exists():
                shutil.copy(src_mask, dst_mask)
        
        print(f"   {split_name}: {len(split_ids)} images + masks")
    
    print(f"\n✅ Organized data saved to {organized_dir}")
    
    # Show example paths
    print("\n" + "=" * 60)
    print("EXAMPLE PATHS FOR TRAINING")
    print("=" * 60)
    print(f"Train image: {organized_dir}/train/images/0001-F-037Y1.jpg")
    print(f"Train mask:  {organized_dir}/train/masks/0001-F-037Y1_mask.png")
    
    return train, val, test

def verify_splits():
    """Verify splits were created correctly"""
    organized_dir = Path("data/processed/organized")
    
    print("\n" + "=" * 60)
    print("VERIFYING SPLITS")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        img_dir = organized_dir / split / "images"
        mask_dir_split = organized_dir / split / "masks"
        
        num_images = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        num_masks = len(list(mask_dir_split.glob("*.png"))) if mask_dir_split.exists() else 0
        
        print(f"\n{split.upper()}:")
        print(f"   Images: {num_images}")
        print(f"   Masks:  {num_masks}")
        
        if num_images == num_masks and num_images > 0:
            print(f"   ✅ Match! Ready for training")
        else:
            print(f"   ❌ Mismatch! Run create_splits.py again")

if __name__ == "__main__":
    create_splits()
    verify_splits()