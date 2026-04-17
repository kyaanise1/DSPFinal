# src/phase2_preprocessing_nnunet.py
"""
Phase 2: BASP Preprocessing for nnUNet v1
Prepares data in nnUNet v1 expected format:

nnUNet_raw_data_base/
└── nnUNet_raw_data/
    ├── Task501_BASP_basp_proposed/
    │   ├── imagesTr/
    │   ├── labelsTr/
    │   └── dataset.json
    ├── Task502_BASP_selective_clahe/
    ├── Task503_BASP_selective_ad/
    └── Task504_BASP_full_non_selective/
"""

import cv2
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json

class BASPPreprocessor:
    """
    Branch-Adaptive Selective Preprocessing (BASP) for nnUNet v1
    """
    
    def __init__(self, clahe_clip_limit=2.0, clahe_grid_size=(8, 8), 
                 ad_iterations=5, ad_kappa=50, ad_gamma=0.1):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.ad_iterations = ad_iterations
        self.ad_kappa = ad_kappa
        self.ad_gamma = ad_gamma
        
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
        """CLAHE followed by Anisotropic Diffusion"""
        return self.apply_anisotropic_diffusion(self.apply_clahe(image))
    
    # ============================================
    # ABLATION METHODS
    # ============================================
    
    def basp_proposed(self, image):
        """BASP: nnUNet = CLAHE+AD, Mask R-CNN = Raw"""
        return self.apply_clahe_ad(image), image.copy()
    
    def selective_clahe(self, image):
        """Selective CLAHE: nnUNet = CLAHE+AD, Mask R-CNN = AD only"""
        return self.apply_clahe_ad(image), self.apply_anisotropic_diffusion(image)
    
    def selective_ad(self, image):
        """Selective AD: nnUNet = CLAHE+AD, Mask R-CNN = CLAHE only"""
        return self.apply_clahe_ad(image), self.apply_clahe(image)
    
    def full_non_selective(self, image):
        """Full: Both branches get CLAHE+AD"""
        processed = self.apply_clahe_ad(image)
        return processed, processed


def convert_to_nifti(image_2d, output_path):
    """
    Convert 2D image to NIfTI format for nnUNet.
    nnUNet expects 3D volumes even for 2D images: (1, H, W, 1)
    """
    # Ensure 2D grayscale
    if len(image_2d.shape) == 3:
        image_2d = cv2.cvtColor(image_2d, cv2.COLOR_BGR2GRAY)
    
    # Add dimensions: (H, W) -> (1, H, W, 1)
    nifti_data = image_2d[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(nifti_data, affine=np.eye(4))
    nib.save(nifti_img, output_path)


def prepare_nnunet_datasets():
    """
    Prepare datasets in nnUNet v1 format for each ablation variant.
    
    CORRECT PATH: nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_{name}/
    """
    
    # Input: Standardized images and masks from Phase 1
    img_dir = Path("data/processed/organized/train/images")
    mask_dir = Path("data/processed/organized/train/masks")
    
    # nnUNet base directory (UPDATED to correct path)
    nnunet_base = Path("nnUNet_raw_data_base/nnUNet_raw_data")
    
    # Initialize preprocessor
    preprocessor = BASPPreprocessor()
    
    # Define 4 ablation variants with Task IDs
    ablations = {
        'basp_proposed': {
            'name': 'BASP (Proposed)',
            'task_id': 501,
            'func': preprocessor.basp_proposed,
        },
        'selective_clahe': {
            'name': 'Selective CLAHE',
            'task_id': 502,
            'func': preprocessor.selective_clahe,
        },
        'selective_ad': {
            'name': 'Selective AD',
            'task_id': 503,
            'func': preprocessor.selective_ad,
        },
        'full_non_selective': {
            'name': 'Full Non-Selective',
            'task_id': 504,
            'func': preprocessor.full_non_selective,
        }
    }
    
    print("=" * 60)
    print("PHASE 2: PREPARING nnUNet v1 DATASETS")
    print("=" * 60)
    print(f"Output base: {nnunet_base.absolute()}")
    
    # Create base directory if not exists
    nnunet_base.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(img_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} training images")
    
    if len(image_files) == 0:
        print("  No images found! Run Phase 1 first.")
        return
    
    # Process each ablation
    for ab_key, ab_info in ablations.items():
        task_id = ab_info['task_id']
        task_name = f"Task{task_id}_BASP_{ab_key}"
        task_dir = nnunet_base / task_name
        
        # Create nnUNet folder structure
        images_tr_dir = task_dir / "imagesTr"
        labels_tr_dir = task_dir / "labelsTr"
        images_tr_dir.mkdir(parents=True, exist_ok=True)
        labels_tr_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔬 {ab_info['name']} → {task_name}")
        print(f"   Location: {task_dir.absolute()}")
        
        # Process each image
        for img_path in tqdm(image_files, desc="   Converting to NIfTI"):
            case_id = img_path.stem
            
            # Load image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_dir / f"{case_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"   Warning: Could not load {case_id}")
                continue
            
            # Apply ablation preprocessing to image
            nnunet_input, _ = ab_info['func'](image)
            
            # Save as NIfTI for nnUNet
            img_nifti_path = images_tr_dir / f"{case_id}_0000.nii.gz"
            mask_nifti_path = labels_tr_dir / f"{case_id}.nii.gz"
            
            convert_to_nifti(nnunet_input, img_nifti_path)
            convert_to_nifti(mask, mask_nifti_path)
        
        # Create dataset.json for this task
        create_dataset_json(task_dir, task_id, ab_key)
        
        # Verify
        num_images = len(list(images_tr_dir.glob("*.nii.gz")))
        num_labels = len(list(labels_tr_dir.glob("*.nii.gz")))
        print(f"     Saved: {num_images} images, {num_labels} labels")
    
    # Print summary
    print("\n" + "=" * 60)
    print("    nnUNet v1 DATASETS READY!")
    print("=" * 60)
    print("\n  Folder structure:")
    print("   nnUNet_raw_data_base/")
    print("   └── nnUNet_raw_data/")
    for ab_key in ablations.keys():
        print(f"       ├── Task{ablations[ab_key]['task_id']}_BASP_{ab_key}/")
        print(f"       │   ├── imagesTr/")
        print(f"       │   ├── labelsTr/")
        print(f"       │   └── dataset.json")
    
    return ablations


def create_dataset_json(task_dir, task_id, variant_name):
    """Create dataset.json file for nnUNet task"""
    
    dataset_json = {
        "name": f"BASP_{variant_name}",
        "description": f"Lumbar spine segmentation - {variant_name}",
        "reference": "BUU-LSPINE dataset",
        "licence": "CC-BY-NC-SA",
        "release": "1.0",
        "tensorImageSize": "4D",
        "modality": {
            "0": "X-ray"
        },
        "labels": {
            "0": "background",
            "1": "L1",
            "2": "L2",
            "3": "L3",
            "4": "L4",
            "5": "L5"
        },
        "numTraining": len(list((task_dir / "imagesTr").glob("*.nii.gz"))),
        "numTest": 0,
        "training": [],
        "test": []
    }
    
    # Add training pairs
    for img_path in sorted((task_dir / "imagesTr").glob("*_0000.nii.gz")):
        case_id = img_path.stem.replace("_0000", "")
        dataset_json["training"].append({
            "image": f"./imagesTr/{img_path.name}",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
    
    with open(task_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"   Created dataset.json with {dataset_json['numTraining']} training cases")


def print_nnunet_commands():
    """Print nnUNet training commands"""
    
    print("\n" + "=" * 60)
    print("🚀 nnUNet v1 TRAINING COMMANDS")
    print("=" * 60)
    print("""
# 1. Set environment variables (run in terminal)
set nnUNet_raw_data_base=C:\\Acads\\3rd Year\\2nd Semester\\DSP\\SPEAR-Net\\SPEAR-Net\\nnUNet_raw_data_base
set nnUNet_preprocessed=C:\\Acads\\3rd Year\\2nd Semester\\DSP\\SPEAR-Net\\SPEAR-Net\\nnunet_preprocessed
set RESULTS_FOLDER=C:\\Acads\\3rd Year\\2nd Semester\\DSP\\SPEAR-Net\\SPEAR-Net\\nnunet_trained_models

# 2. Run preprocessing for each task
nnUNet_plan_and_preprocess -t 501
nnUNet_plan_and_preprocess -t 502
nnUNet_plan_and_preprocess -t 503
nnUNet_plan_and_preprocess -t 504

# 3. Train 5-fold cross-validation for Task 501 (BASP Proposed)
nnUNet_train 2d nnUNetTrainerV2 501 0
nnUNet_train 2d nnUNetTrainerV2 501 1
nnUNet_train 2d nnUNetTrainerV2 501 2
nnUNet_train 2d nnUNetTrainerV2 501 3
nnUNet_train 2d nnUNetTrainerV2 501 4

# 4. Repeat for other tasks (502, 503, 504)

# 5. Run inference
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 501 -m 2d
""")


def visualize_ablation():
    """Quick visualization of preprocessing differences"""
    
    preprocessor = BASPPreprocessor()
    
    # Load sample
    sample_path = Path("data/standardized/images/0001-F-037Y1.jpg")
    
    if not sample_path.exists():
        print("Sample not found for visualization")
        return
    
    image = cv2.imread(str(sample_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply ablations
    basp_nnunet, basp_rcnn = preprocessor.basp_proposed(image)
    sel_clahe_nnunet, sel_clahe_rcnn = preprocessor.selective_clahe(image)
    full_nnunet, full_rcnn = preprocessor.full_non_selective(image)
    
    # Convert to RGB
    basp_nnunet_rgb = cv2.cvtColor(basp_nnunet, cv2.COLOR_BGR2RGB)
    basp_rcnn_rgb = cv2.cvtColor(basp_rcnn, cv2.COLOR_BGR2RGB)
    sel_clahe_rcnn_rgb = cv2.cvtColor(sel_clahe_rcnn, cv2.COLOR_BGR2RGB)
    full_rcnn_rgb = cv2.cvtColor(full_rcnn, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(basp_nnunet_rgb, cmap='gray')
    axes[0, 0].set_title("nnUNet Input\n(CLAHE+AD)\nSame for ALL")
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(basp_rcnn_rgb, cmap='gray')
    axes[1, 0].set_title("BASP (Proposed)\nMask R-CNN: Raw")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sel_clahe_rcnn_rgb, cmap='gray')
    axes[1, 1].set_title("Selective CLAHE\nMask R-CNN: AD only")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(full_rcnn_rgb, cmap='gray')
    axes[1, 2].set_title("Full Non-Selective\nMask R-CNN: CLAHE+AD")
    axes[1, 2].axis('off')
    
    plt.suptitle("BASP Ablation: nnUNet v1 + Mask R-CNN", fontsize=14)
    plt.tight_layout()
    plt.savefig("basp_nnunet_ablation.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Visualization saved as 'basp_nnunet_ablation.png'")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("BASP: nnUNet v1 Dataset Preparation")
    print("=" * 60)
    
    # Visualize
    visualize_ablation()
    
    # Prepare datasets
    prepare_nnunet_datasets()
    
    # Print training commands
    print_nnunet_commands()
    
    print("\n  Phase 2 Complete!")
    print("    Next: Set environment variables and run nnUNet training")