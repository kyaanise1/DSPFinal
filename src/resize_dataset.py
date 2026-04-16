# src/resize_dataset.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def resize_with_padding(image, target_size=(256, 256), padding=3):
    """
    Resize image to target size with padding.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        target_size: (height, width) target size
        padding: Padding size in pixels (adds padding around resized image)
    
    Returns:
        resized: Image with padding applied
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate resize ratio to fit inside target (preserving aspect ratio)
    ratio = min(target_h / h, target_w / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with padding
    canvas_h = target_h + (2 * padding)
    canvas_w = target_w + (2 * padding)
    
    # Initialize canvas (grayscale or color)
    if len(image.shape) == 3:
        canvas = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (canvas_h - new_h) // 2
    x_offset = (canvas_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Final resize to exact target size (removes padding if needed)
    final = cv2.resize(canvas, target_size)
    
    return final

def pad_mask(mask, target_size=(256, 256), padding=3):
    """
    Pad mask with zeros (background) using NEAREST interpolation.
    Preserves class ID values.
    """
    h, w = mask.shape[:2]
    target_h, target_w = target_size
    
    # Calculate resize ratio
    ratio = min(target_h / h, target_w / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    
    # Resize mask (using NEAREST to preserve class IDs)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create canvas with padding (background = 0)
    canvas_h = target_h + (2 * padding)
    canvas_w = target_w + (2 * padding)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Center the resized mask on canvas
    y_offset = (canvas_h - new_h) // 2
    x_offset = (canvas_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Final resize to exact target size
    final = cv2.resize(canvas, target_size, interpolation=cv2.INTER_NEAREST)
    
    return final

def resize_dataset(target_size=(256, 256), padding=3):
    """
    Resize all images and masks to target size with padding.
    This is a ONE-TIME standardization step before Phase 2.
    """
    
    # Paths
    original_images_dir = Path("data/raw/images")
    original_masks_dir = Path("data/processed/training_masks")
    
    # Output paths for standardized data
    output_images_dir = Path("data/standardized/images")
    output_masks_dir = Path("data/standardized/masks")
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("IMAGE STANDARDIZATION (RESIZE + PADDING)")
    print("=" * 60)
    print(f"Target size: {target_size}")
    print(f"Padding: {padding}px (3x3)")
    print(f"Original images: {original_images_dir}")
    print(f"Original masks: {original_masks_dir}")
    print(f"Output images: {output_images_dir}")
    print(f"Output masks: {output_masks_dir}")
    
    # Get all images
    image_files = list(original_images_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    # Resize images
    print("\n📁 Resizing images with padding...")
    for img_path in tqdm(image_files, desc="  Images"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        resized = resize_with_padding(img, target_size, padding)
        output_path = output_images_dir / img_path.name
        cv2.imwrite(str(output_path), resized)
    
    # Resize masks (using NEAREST for class IDs)
    mask_files = list(original_masks_dir.glob("*_mask.png"))
    print(f"\n📁 Resizing masks with padding (NEAREST interpolation)...")
    
    for mask_path in tqdm(mask_files, desc="  Masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        resized = pad_mask(mask, target_size, padding)
        output_path = output_masks_dir / mask_path.name
        cv2.imwrite(str(output_path), resized)
    
    # Verify
    verify_resized(output_images_dir, output_masks_dir, target_size)
    
    print("\n" + "=" * 60)
    print("✅ IMAGE STANDARDIZATION COMPLETE!")
    print(f"   Resized images: {output_images_dir}")
    print(f"   Resized masks: {output_masks_dir}")
    print(f"   Size: {target_size[0]}x{target_size[1]} with {padding}px padding")
    print("=" * 60)


def verify_resized(images_dir, masks_dir, target_size=(256, 256)):
    """Verify resized images and masks"""
    
    sample_img = list(images_dir.glob("*.jpg"))[0]
    sample_mask = list(masks_dir.glob("*_mask.png"))[0]
    
    img = cv2.imread(str(sample_img))
    mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
    
    print("\n🔍 VERIFICATION:")
    print(f"   Image shape: {img.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Expected: {target_size[0]}x{target_size[1]}")
    print(f"   Mask unique values: {np.unique(mask)}")
    
    if img.shape[:2] == (target_size[0], target_size[1]):
        print("   ✅ Image size matches target!")
    else:
        print(f"   ❌ Image size mismatch! Got {img.shape[:2]}, expected {target_size}")
    
    if mask.shape[:2] == (target_size[0], target_size[1]):
        print("   ✅ Mask size matches target!")
    else:
        print(f"   ❌ Mask size mismatch! Got {mask.shape[:2]}, expected {target_size}")


def update_create_splits():
    """Reminder to update create_splits.py"""
    
    print("\n" + "=" * 60)
    print("⚠️ IMPORTANT: Update your create_splits.py")
    print("=" * 60)
    print("Change these paths in create_splits.py:")
    print("")
    print("  FROM:")
    print("    original_image_dir = Path('data/raw/images')")
    print("    mask_dir = Path('data/processed/training_masks')")
    print("")
    print("  TO:")
    print("    original_image_dir = Path('data/standardized/images')")
    print("    mask_dir = Path('data/standardized/masks')")
    print("")
    print("Then re-run: python src/create_splits.py")


def preview_standardization():
    """Preview original vs standardized image"""
    
    # Find sample files
    original_img_dir = Path("data/raw/images")
    standardized_img_dir = Path("data/standardized/images")
    
    if not original_img_dir.exists() or not standardized_img_dir.exists():
        print("Run resize_dataset() first")
        return
    
    sample = list(original_img_dir.glob("*.jpg"))[0]
    original = cv2.imread(str(sample))
    standardized = cv2.imread(str(standardized_img_dir / sample.name))
    
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    standardized_rgb = cv2.cvtColor(standardized, cv2.COLOR_BGR2RGB)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_rgb, cmap='gray')
    axes[0].set_title(f"Original\n{original.shape[:2]}")
    axes[0].axis('off')
    
    axes[1].imshow(standardized_rgb, cmap='gray')
    axes[1].set_title(f"Standardized (256x256 + 3px padding)\n{standardized.shape[:2]}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("standardization_preview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Preview saved as 'standardization_preview.png'")


if __name__ == "__main__":
    # Step 1: Resize dataset
    resize_dataset(target_size=(256, 256), padding=3)
    
    # Step 2: Preview result
    preview_standardization()
    
    # Step 3: Reminder to update splits
    update_create_splits()