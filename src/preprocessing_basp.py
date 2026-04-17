# src/phase2_preprocessing.py
"""
Phase 2: BASP Preprocessing Ablation Experiments
Applies different preprocessing combinations to U-Net and Mask R-CNN branches

Image Standardization (resize + padding) is already done in Phase 1.
This script applies CLAHE and/or Anisotropic Diffusion as per ablation setup.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class BASPPreprocessor:
    """
    Branch-Adaptive Selective Preprocessing (BASP)
    Implements 4 ablation variants for VertXNet ensemble
    """
    
    def __init__(self, clahe_clip_limit=2.0, clahe_grid_size=(8, 8), 
                 ad_iterations=5, ad_kappa=50, ad_gamma=0.1):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.ad_iterations = ad_iterations
        self.ad_kappa = ad_kappa
        self.ad_gamma = ad_gamma
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )
    
    def apply_clahe(self, image):
        """Apply CLAHE contrast enhancement"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        enhanced = self.clahe.apply(gray)
        
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def apply_anisotropic_diffusion(self, image):
        """Apply edge-preserving denoising (Perona-Malik)"""
        if len(image.shape) == 3:
            img_float = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            img_float = image.astype(np.float32)
        
        for _ in range(self.ad_iterations):
            dx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
            
            c_dx = np.exp(-(dx / self.ad_kappa) ** 2)
            c_dy = np.exp(-(dy / self.ad_kappa) ** 2)
            
            img_float = img_float + self.ad_gamma * (c_dx * dx + c_dy * dy)
        
        result = np.clip(img_float, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def apply_clahe_ad(self, image):
        """Apply CLAHE followed by Anisotropic Diffusion"""
        return self.apply_anisotropic_diffusion(self.apply_clahe(image))
    
    # ============================================
    # BASP (PROPOSED)
    # ============================================
    def basp_proposed(self, image):
        """
        Proposed BASP Method:
        - U-Net: CLAHE + AD
        - Mask R-CNN: Raw (no preprocessing)
        """
        unet_input = self.apply_clahe_ad(image)
        maskrcnn_input = image.copy()
        return unet_input, maskrcnn_input
    
    # ============================================
    # SELECTIVE CLAHE
    # ============================================
    def selective_clahe(self, image):
        """
        Selective CLAHE:
        - U-Net: CLAHE + AD
        - Mask R-CNN: AD only
        """
        unet_input = self.apply_clahe_ad(image)
        maskrcnn_input = self.apply_anisotropic_diffusion(image)
        return unet_input, maskrcnn_input
    
    # ============================================
    # SELECTIVE AD
    # ============================================
    def selective_ad(self, image):
        """
        Selective Anisotropic Diffusion:
        - U-Net: CLAHE + AD
        - Mask R-CNN: CLAHE only
        """
        unet_input = self.apply_clahe_ad(image)
        maskrcnn_input = self.apply_clahe(image)
        return unet_input, maskrcnn_input
    
    # ============================================
    # FULL NON-SELECTIVE
    # ============================================
    def full_non_selective(self, image):
        """
        Full Non-Selective:
        - U-Net: CLAHE + AD
        - Mask R-CNN: CLAHE + AD
        """
        processed = self.apply_clahe_ad(image)
        return processed, processed


def run_ablation_experiments():
    """
    Apply all 4 ablation variants to the training dataset.
    Input: Standardized images (256x256 + padding) from Phase 1
    Output: Preprocessed images for each ablation variant
    """
    
    # Input: Standardized images from Phase 1
    input_dir = Path("data/processed/organized/train/images")
    
    # Output: Preprocessed images for each ablation
    output_base = Path("data/processed/ablation")
    
    print("=" * 60)
    print("PHASE 2: BASP PREPROCESSING ABLATION EXPERIMENTS")
    print("=" * 60)
    print(f"Input (standardized images): {input_dir}")
    print(f"Output base: {output_base}")
    
    # Check input directory
    if not input_dir.exists():
        print(f"\n Input directory not found: {input_dir}")
        print("   Make sure Phase 1 (resize + splits) is complete.")
        return
    
    # Initialize preprocessor
    preprocessor = BASPPreprocessor()
    
    # Define 4 ablation variants with CLEAN NAMES
    ablations = {
        'basp_proposed': {
            'name': 'BASP (Proposed)',
            'func': preprocessor.basp_proposed,
            'unet_desc': 'CLAHE + AD',
            'rcnn_desc': 'Raw (No preprocessing)'
        },
        'selective_clahe': {
            'name': 'Selective CLAHE',
            'func': preprocessor.selective_clahe,
            'unet_desc': 'CLAHE + AD',
            'rcnn_desc': 'AD only'
        },
        'selective_ad': {
            'name': 'Selective AD',
            'func': preprocessor.selective_ad,
            'unet_desc': 'CLAHE + AD',
            'rcnn_desc': 'CLAHE only'
        },
        'full_non_selective': {
            'name': 'Full Non-Selective',
            'func': preprocessor.full_non_selective,
            'unet_desc': 'CLAHE + AD',
            'rcnn_desc': 'CLAHE + AD'
        }
    }
    
    # Get all training images
    image_files = list(input_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} training images")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # Process each ablation variant
    print("\n📊 Processing Ablation Variants...")
    print("-" * 60)
    
    for ab_key, ab_info in ablations.items():
        # Create output directories with CLEAN NAMES
        unet_dir = output_base / ab_key / "unet_input"
        rcnn_dir = output_base / ab_key / "maskrcnn_input"
        unet_dir.mkdir(parents=True, exist_ok=True)
        rcnn_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔬 {ab_info['name']}:")
        print(f"   U-Net input: {ab_info['unet_desc']}")
        print(f"   Mask R-CNN input: {ab_info['rcnn_desc']}")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"   Processing"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Apply ablation preprocessing
            unet_input, rcnn_input = ab_info['func'](image)
            
            # Save
            unet_path = unet_dir / img_path.name
            rcnn_path = rcnn_dir / img_path.name
            
            cv2.imwrite(str(unet_path), unet_input)
            cv2.imwrite(str(rcnn_path), rcnn_input)
        
        # Count saved files
        unet_count = len(list(unet_dir.glob("*.jpg")))
        rcnn_count = len(list(rcnn_dir.glob("*.jpg")))
        print(f"    Saved: {unet_count} U-Net + {rcnn_count} Mask R-CNN images")
    
    # Print summary
    print("\n" + "=" * 60)
    print(" PHASE 2 COMPLETE!")
    print("=" * 60)
    print("\n Output structure:")
    print("   data/processed/ablation/")
    print("   ├── basp_proposed/")
    print("   │   ├── unet_input/")
    print("   │   └── maskrcnn_input/")
    print("   ├── selective_clahe/")
    print("   │   ├── unet_input/")
    print("   │   └── maskrcnn_input/")
    print("   ├── selective_ad/")
    print("   │   ├── unet_input/")
    print("   │   └── maskrcnn_input/")
    print("   └── full_non_selective/")
    print("       ├── unet_input/")
    print("       └── maskrcnn_input/")
    
    return ablations


def visualize_ablation_comparison():
    """Visualize all 4 ablation variants side by side"""
    
    preprocessor = BASPPreprocessor()
    
    # Load a sample standardized image
    sample_path = Path("data/standardized/images/0001-F-037Y1.jpg")
    
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return
    
    original = cv2.imread(str(sample_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Apply all ablations
    basp_unet, basp_rcnn = preprocessor.basp_proposed(original)
    sel_clahe_unet, sel_clahe_rcnn = preprocessor.selective_clahe(original)
    sel_ad_unet, sel_ad_rcnn = preprocessor.selective_ad(original)
    full_unet, full_rcnn = preprocessor.full_non_selective(original)
    
    # Convert to RGB
    basp_unet_rgb = cv2.cvtColor(basp_unet, cv2.COLOR_BGR2RGB)
    basp_rcnn_rgb = cv2.cvtColor(basp_rcnn, cv2.COLOR_BGR2RGB)
    sel_clahe_rcnn_rgb = cv2.cvtColor(sel_clahe_rcnn, cv2.COLOR_BGR2RGB)
    sel_ad_rcnn_rgb = cv2.cvtColor(sel_ad_rcnn, cv2.COLOR_BGR2RGB)
    full_rcnn_rgb = cv2.cvtColor(full_rcnn, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: U-Net inputs (all identical - CLAHE+AD)
    axes[0, 0].imshow(basp_unet_rgb, cmap='gray')
    axes[0, 0].set_title("U-Net Input\n(CLAHE + AD)\nSame for ALL ablations", fontsize=11)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    # Bottom row: Mask R-CNN inputs (varies by ablation)
    axes[1, 0].imshow(basp_rcnn_rgb, cmap='gray')
    axes[1, 0].set_title("BASP (Proposed)\nMask R-CNN: Raw", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sel_clahe_rcnn_rgb, cmap='gray')
    axes[1, 1].set_title("Selective CLAHE\nMask R-CNN: AD only", fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(full_rcnn_rgb, cmap='gray')
    axes[1, 2].set_title("Full Non-Selective\nMask R-CNN: CLAHE+AD", fontsize=11)
    axes[1, 2].axis('off')
    
    plt.suptitle("BASP Ablation Study: Mask R-CNN Branch Variations\n(U-Net branch fixed: CLAHE+AD for all)", fontsize=14)
    plt.tight_layout()
    plt.savefig("basp_ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'basp_ablation_comparison.png'")


def print_summary_table():
    """Print ablation summary table"""
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY TABLE")
    print("=" * 60)
    print("\n| # | Ablation Name          | U-Net Input    | Mask R-CNN Input    |")
    print("|---|------------------------|----------------|---------------------|")
    print("| 1 | BASP (Proposed)        | CLAHE + AD     | Raw (—)             |")
    print("| 2 | Selective CLAHE        | CLAHE + AD     | AD only             |")
    print("| 3 | Selective AD           | CLAHE + AD     | CLAHE only          |")
    print("| 4 | Full Non-Selective     | CLAHE + AD     | CLAHE + AD          |")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BASP: BRANCH-ADAPTIVE SELECTIVE PREPROCESSING")
    print("PHASE 2 - PREPROCESSING ABLATION EXPERIMENTS")
    print("=" * 60)
    
    # Print summary table
    print_summary_table()
    
    # Visualize comparison
    visualize_ablation_comparison()
    
    # Run ablation experiments
    run_ablation_experiments()
    
    print("\n Phase 2 Complete!")