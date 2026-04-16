# src/basp_pipeline.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class BASPPipeline:
    """
    Branch-Adaptive Selective Preprocessing (BASP) Pipeline
    For multiclass lumbar vertebral segmentation
    """
    
    def __init__(self, target_size=(512, 512)):
        """
        Args:
            target_size: (height, width) for image standardization
        """
        self.target_size = target_size
        
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Anisotropic Diffusion parameters
        self.ad_iterations = 5
        self.ad_kappa = 50
        self.ad_gamma = 0.1
    
    # ============================================
    # STEP 1: IMAGE STANDARDIZATION
    # ============================================
    
    def standardize(self, image):
        """
        Image Standardization:
        - Resize to fixed dimensions
        - Normalize intensity (0-255 range)
        """
        # Resize
        resized = cv2.resize(image, self.target_size)
        
        # Normalize intensity (if needed)
        if resized.max() > 0:
            normalized = (resized / resized.max() * 255).astype(np.uint8)
        else:
            normalized = resized
        
        return normalized
    
    # ============================================
    # STEP 2: PREPROCESSING (BASP - Branch Specific)
    # ============================================
    
    def apply_clahe(self, image):
        """Apply CLAHE contrast enhancement"""
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convert back to 3-channel
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def apply_anisotropic_diffusion(self, image):
        """Apply edge-preserving denoising"""
        # Convert to grayscale if color
        if len(image.shape) == 3:
            img_float = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            img_float = image.astype(np.float32)
        
        # Perona-Malik diffusion
        for _ in range(self.ad_iterations):
            dx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
            
            c_dx = np.exp(-(dx / self.ad_kappa) ** 2)
            c_dy = np.exp(-(dy / self.ad_kappa) ** 2)
            
            img_float = img_float + self.ad_gamma * (c_dx * dx + c_dy * dy)
        
        # Convert back
        result = np.clip(img_float, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def apply_clahe_ad(self, image):
        """CLAHE followed by Anisotropic Diffusion"""
        return self.apply_anisotropic_diffusion(self.apply_clahe(image))
    
    # ============================================
    # BRANCH-SPECIFIC PREPROCESSING (BASP)
    # ============================================
    
    def preprocess_u_net(self, image):
        """
        U-Net branch preprocessing:
        Standardize → CLAHE → Anisotropic Diffusion
        """
        standardized = self.standardize(image)
        return self.apply_clahe_ad(standardized)
    
    def preprocess_mask_rcnn(self, image):
        """
        Mask R-CNN branch preprocessing:
        Standardize only (Raw)
        """
        return self.standardize(image)
    
    # ============================================
    # STEP 3: ENSEMBLE MODULE
    # ============================================
    
    def ensemble_module(self, unet_masks, maskrcnn_masks):
        """
        Ensemble Module with:
        1. Agreement matching
        2. Overlap resolution
        3. Missed vertebra recovery
        4. Mask selection and sorting
        
        Args:
            unet_masks: List of masks from U-Net (semantic)
            maskrcnn_masks: List of masks from Mask R-CNN (instance)
        
        Returns:
            final_masks: List of merged, sorted masks with vertebra labels
        """
        
        # Step 1: Agreement matching
        # Match U-Net and Mask R-CNN masks by centroid distance
        matched_pairs = self.match_masks(unet_masks, maskrcnn_masks)
        
        # Step 2: Overlap resolution
        # Resolve overlapping masks (keep higher confidence)
        resolved_masks = self.resolve_overlaps(matched_pairs)
        
        # Step 3: Missed vertebra recovery
        # If a vertebra is detected by only one model, include it
        recovered_masks = self.recover_missed(unet_masks, maskrcnn_masks, resolved_masks)
        
        # Step 4: Mask selection and sorting
        # Sort by anatomical order (top to bottom: L1→L5)
        final_masks = self.sort_masks(recovered_masks)
        
        return final_masks
    
    def match_masks(self, unet_masks, maskrcnn_masks):
        """Match masks by centroid distance"""
        matched = []
        
        for unet_mask in unet_masks:
            unet_centroid = self.get_centroid(unet_mask)
            best_match = None
            best_distance = float('inf')
            
            for rcnn_mask in maskrcnn_masks:
                rcnn_centroid = self.get_centroid(rcnn_mask)
                distance = np.linalg.norm(unet_centroid - rcnn_centroid)
                
                if distance < best_distance and distance < 50:  # Threshold
                    best_distance = distance
                    best_match = rcnn_mask
            
            if best_match is not None:
                matched.append((unet_mask, best_match))
        
        return matched
    
    def resolve_overlaps(self, matched_pairs):
        """Resolve overlapping masks"""
        resolved = []
        
        for unet_mask, rcnn_mask in matched_pairs:
            # Combine masks (agreement = keep both)
            combined = cv2.bitwise_or(unet_mask, rcnn_mask)
            resolved.append(combined)
        
        return resolved
    
    def recover_missed(self, unet_masks, maskrcnn_masks, resolved):
        """Recover vertebrae missed by one model"""
        recovered = resolved.copy()
        
        # Get centroids of already resolved masks
        existing_centroids = [self.get_centroid(m) for m in resolved]
        
        # Check U-Net masks not yet included
        for mask in unet_masks:
            centroid = self.get_centroid(mask)
            if not self.is_duplicate(centroid, existing_centroids):
                recovered.append(mask)
        
        # Check Mask R-CNN masks not yet included
        for mask in maskrcnn_masks:
            centroid = self.get_centroid(mask)
            if not self.is_duplicate(centroid, existing_centroids):
                recovered.append(mask)
        
        return recovered
    
    def sort_masks(self, masks):
        """Sort masks by anatomical order (top to bottom = L1 to L5)"""
        # Get centroids and sort by y-coordinate
        with_centroids = [(m, self.get_centroid(m)[1]) for m in masks]
        sorted_masks = sorted(with_centroids, key=lambda x: x[1])  # Sort by y
        
        # Assign vertebra labels (L1 to L5)
        vertebra_labels = ['L1', 'L2', 'L3', 'L4', 'L5']
        final_masks = []
        
        for idx, (mask, y) in enumerate(sorted_masks[:5]):
            final_masks.append({
                'mask': mask,
                'label': vertebra_labels[idx] if idx < len(vertebra_labels) else f'V{idx+1}',
                'centroid_y': y
            })
        
        return final_masks
    
    def get_centroid(self, mask):
        """Calculate centroid of a binary mask"""
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return np.array([cx, cy])
        return np.array([0, 0])
    
    def is_duplicate(self, centroid, existing_centroids, threshold=30):
        """Check if centroid is too close to existing ones"""
        for existing in existing_centroids:
            if np.linalg.norm(centroid - existing) < threshold:
                return True
        return False
    
    # ============================================
    # STEP 4: VERTEBRA IDENTIFICATION
    # ============================================
    
    def identify_vertebrae(self, final_masks):
        """
        Vertebra Identification using reference vertebra (S1 anchor)
        
        Since BUU-LSPINE(400) has no S1, we use L1 as reference
        and label sequentially down to L5
        """
        # Sort by y-coordinate (top to bottom)
        sorted_masks = sorted(final_masks, key=lambda x: x['centroid_y'])
        
        # Assign labels sequentially
        vertebra_names = ['L1', 'L2', 'L3', 'L4', 'L5']
        
        for idx, item in enumerate(sorted_masks[:5]):
            item['label'] = vertebra_names[idx]
        
        return sorted_masks
    
    # ============================================
    # FULL PIPELINE
    # ============================================
    
    def process(self, image):
        """
        Run full BASP pipeline on a single image
        
        Args:
            image: Raw X-ray image
        
        Returns:
            final_segmentation: Labeled vertebral masks
        """
        # Step 1 & 2: Branch-specific preprocessing
        unet_input = self.preprocess_u_net(image)
        maskrcnn_input = self.preprocess_mask_rcnn(image)
        
        # Step 3: Segmentation (placeholder - actual models to be implemented)
        # unet_masks = self.u_net.predict(unet_input)
        # maskrcnn_masks = self.mask_rcnn.predict(maskrcnn_input)
        
        # Placeholder for now
        unet_masks = []
        maskrcnn_masks = []
        
        # Step 4: Ensemble
        ensemble_masks = self.ensemble_module(unet_masks, maskrcnn_masks)
        
        # Step 5: Vertebra identification
        final_result = self.identify_vertebrae(ensemble_masks)
        
        return final_result


def visualize_basp_pipeline():
    """Visualize the BASP pipeline steps on a sample image"""
    
    pipeline = BASPPipeline()
    
    # Load sample image
    sample_path = Path("data/raw/images/0001-F-037Y1.jpg")
    
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return
    
    image = cv2.imread(str(sample_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply each step
    standardized = pipeline.standardize(image)
    standardized_rgb = cv2.cvtColor(standardized, cv2.COLOR_BGR2RGB)
    
    unet_input = pipeline.preprocess_u_net(image)
    unet_input_rgb = cv2.cvtColor(unet_input, cv2.COLOR_BGR2RGB)
    
    maskrcnn_input = pipeline.preprocess_mask_rcnn(image)
    maskrcnn_input_rgb = cv2.cvtColor(maskrcnn_input, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image_rgb, cmap='gray')
    axes[0, 0].set_title("Step 1: Raw X-ray")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(standardized_rgb, cmap='gray')
    axes[0, 1].set_title("Step 2: Standardized\n(Resize + Normalize)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(unet_input_rgb, cmap='gray')
    axes[1, 0].set_title("Step 3: U-Net Branch\n(CLAHE + Anisotropic Diffusion)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(maskrcnn_input_rgb, cmap='gray')
    axes[1, 1].set_title("Step 3: Mask R-CNN Branch\n(Raw - No Enhancement)")
    axes[1, 1].axis('off')
    
    plt.suptitle("BASP Pipeline: Branch-Adaptive Selective Preprocessing", fontsize=14)
    plt.tight_layout()
    plt.savefig("basp_pipeline_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ BASP Pipeline visualization saved as 'basp_pipeline_visualization.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("BASP: BRANCH-ADAPTIVE SELECTIVE PREPROCESSING PIPELINE")
    print("=" * 60)
    
    visualize_basp_pipeline()
    
    print("\n📋 BASP Pipeline Summary:")
    print("   1. Image Standardization (Resize + Normalize)")
    print("   2. U-Net Branch: CLAHE → Anisotropic Diffusion")
    print("   3. Mask R-CNN Branch: Raw (No enhancement)")
    print("   4. Ensemble Module (Agreement + Overlap + Recovery + Sorting)")
    print("   5. Vertebra Identification (L1-L5 labeling)")