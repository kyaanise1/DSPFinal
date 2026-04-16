# src/create_splits.py (UPDATED FOR STANDARDIZED DATA)
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def create_splits():
    """Create train/val/test splits with exact counts"""
    
    # ============================================
    # UPDATED PATHS - USING STANDARDIZED DATA
    # ============================================
    original_image_dir = Path("data/standardized/images")   # ← CHANGED
    mask_dir = Path("data/standardized/masks")              # ← CHANGED
    
    print("=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    print(f"Image directory: {original_image_dir}")
    print(f"Mask directory: {mask_dir}")
    
    # Check if directories exist
    if not original_image_dir.exists():
        print(f"\n❌ ERROR: {original_image_dir} not found!")
        print("   Run resize_dataset.py first!")
        return
    
    if not mask_dir.exists():
        print(f"\n❌ ERROR: {mask_dir} not found!")
        print("   Run resize_dataset.py first!")
        return
    
    # Get all mask files
    mask_files = list(mask_dir.glob("*_mask.png"))
    image_ids = [f.stem.replace("_mask", "") for f in mask_files]
    
    total = len(image_ids)
    print(f"\nTotal images found: {total}")
    
    if total == 0:
        print("No masks found!")
        return
    
    # Calculate exact counts
    test_count = int(total * 0.15)  # 60 for 400
    val_count = int(total * 0.15)   # 60 for 400
    train_count = total - test_count - val_count  # 280 for 400
    
    print(f"\nTarget splits:")
    print(f"  Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  Val:   {val_count} ({val_count/total*100:.1f}%)")
    print(f"  Test:  {test_count} ({test_count/total*100:.1f}%)")
    
    # First split: separate test set
    train_val, test = train_test_split(
        image_ids, 
        test_size=test_count,
        random_state=42
    )
    
    # Second split: separate val from train
    train, val = train_test_split(
        train_val,
        test_size=val_count,
        random_state=42
    )
    
    print(f"\nActual splits:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    
    # Save split lists
    split_dir = Path("data/processed/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    with open(split_dir / "train.txt", "w") as f:
        f.write("\n".join(train))
    with open(split_dir / "val.txt", "w") as f:
        f.write("\n".join(val))
    with open(split_dir / "test.txt", "w") as f:
        f.write("\n".join(test))
    
    print(f"\n✓ Split lists saved to {split_dir}")
    
    # Create organized folders
    organized_dir = Path("data/processed/organized")
    
    # Remove existing organized folder if it exists
    if organized_dir.exists():
        shutil.rmtree(organized_dir)
    
    print(f"\n📁 Creating organized folders...")
    
    for split_name, split_ids in [('train', train), ('val', val), ('test', test)]:
        split_img_dir = organized_dir / split_name / "images"
        split_mask_dir = organized_dir / split_name / "masks"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in split_ids:
            # Copy original image (from standardized folder)
            src_img = original_image_dir / f"{img_id}.jpg"
            dst_img = split_img_dir / f"{img_id}.jpg"
            if src_img.exists():
                shutil.copy(src_img, dst_img)
            else:
                print(f"   Warning: {src_img} not found")
            
            # Copy mask (from standardized folder)
            src_mask = mask_dir / f"{img_id}_mask.png"
            dst_mask = split_mask_dir / f"{img_id}_mask.png"
            if src_mask.exists():
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"   Warning: {src_mask} not found")
        
        print(f"   {split_name}: {len(split_ids)} images + masks")
    
    print(f"\n✅ Organized data saved to {organized_dir}")
    
    # Verify
    verify_splits(organized_dir)
    
    return train, val, test

def verify_splits(organized_dir):
    """Verify splits were created correctly"""
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
            print(f"   ❌ Mismatch!")

def check_image_sizes():
    """Check that all images are 256x256"""
    organized_dir = Path("data/processed/organized")
    
    if not organized_dir.exists():
        print("Run create_splits() first")
        return
    
    print("\n" + "=" * 60)
    print("CHECKING IMAGE SIZES")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        img_dir = organized_dir / split / "images"
        if img_dir.exists():
            sample = list(img_dir.glob("*.jpg"))[0]
            import cv2
            img = cv2.imread(str(sample))
            print(f"{split}: {img.shape[:2]} (expected 256x256)")

if __name__ == "__main__":
    create_splits()
    check_image_sizes()