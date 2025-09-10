import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

# We assume the Rect and FixedQuadTree classes from your provided code are available.
# This file can be imported into your main script.

class HDEProcessor:
    """
    Encapsulates the logic for the HDE pre-training task as described in the project plan.
    This class takes the adaptive patches from a FixedQuadTree and generates the final
    mixed sequence of clean and noised patches for the ViT encoder.
    """
    def __init__(self, visible_fraction=0.25):
        """
        Initializes the HDE processor.
        
        Args:
            visible_fraction (float): The fraction of patches that should remain clean (visible).
                                      The rest will be noised.
        """
        assert 0.0 < visible_fraction < 1.0, "visible_fraction must be between 0 and 1."
        self.visible_fraction = visible_fraction

    def generate_noise_for_patch(self, patch):
        """
        Generates content-aware Gaussian noise for a single image patch.
        Noise variance is equal to the patch's average pixel value.
        
        Args:
            patch (np.array): The input image patch.
            
        Returns:
            np.array: A Gaussian noise tensor with the same shape as the patch.
        """
        # Ensure patch is float for accurate mean calculation
        patch_float = patch.astype(np.float32)
        
        # Calculate average pixel value. 
        avg_p = np.mean(patch_float)
        if avg_p <= 0:
            return np.zeros_like(patch_float) # Return zero noise for black patches

        # Standard deviation is the square root of the variance (avg_p)
        std_dev = np.sqrt(avg_p)
        
        # Generate Gaussian noise
        noise = np.random.normal(loc=0.0, scale=std_dev, size=patch.shape)
        
        return noise

    def create_training_sequence(self, img, qdt):
        """
        Creates the final training sequence for the HDE model.
        
        This method implements Step 3 from your project plan: "Partially Visible Guided Denoising".
        It returns a list of patches, where a fraction are the original clean patches
        and the rest have hierarchical noise added.
        
        Args:
            img (np.array): The original full-size image (e.g., 512x512x3).
            qdt (FixedQuadTree): The quadtree object already built on the image's edge map.
            
        Returns:
            list: A list containing patches (np.array), ready for further processing
                  (e.g., resizing, flattening) before being fed to the ViT encoder.
            np.array: A mask indicating which patches were noised (1) and which were visible (0).
        """
        # Get all leaf nodes (Rect objects) from the quadtree
        leaf_nodes = [node for node, value in qdt.nodes]
        num_patches = len(leaf_nodes)
        
        # Create a list of indices to shuffle for random masking
        indices = list(range(num_patches))
        random.shuffle(indices)
        
        # Determine the split point for visible vs. noised patches
        num_visible = int(num_patches * self.visible_fraction)
        
        # Create a set of indices for fast lookups
        visible_indices = set(indices[:num_visible])
        
        final_patches_sequence = []
        is_noised_mask = np.zeros(qdt.fixed_length, dtype=int)

        # Iterate through the original, ordered leaf nodes
        for i, bbox in enumerate(leaf_nodes):
            # Extract the original clean patch
            original_patch = bbox.get_area(img)
            
            if i in visible_indices:
                # This is a visible patch, keep it clean
                final_patches_sequence.append(original_patch)
                is_noised_mask[i] = 0 # 0 for visible
            else:
                # This is a patch to be noised
                is_noised_mask[i] = 1 # 1 for noised
                
                # 1. Generate the content-aware noise
                noise = self.generate_noise_for_patch(original_patch)
                
                # 2. Add noise to the patch
                noised_patch_float = original_patch.astype(np.float32) + noise
                
                # 3. Clip the values to maintain a valid image range
                if img.dtype == np.uint8:
                     noised_patch = np.clip(noised_patch_float, 0, 255).astype(np.uint8)
                else: # Assumes float images scaled to 0-1 or other ranges
                     noised_patch = noised_patch_float

                final_patches_sequence.append(noised_patch)
        
        # Your qdt.serialize method already handles padding, but this ensures the mask is also padded.
        # This part is important if the quadtree generates fewer nodes than fixed_length.
        padding_needed = qdt.fixed_length - len(final_patches_sequence)
        if padding_needed > 0:
            is_noised_mask[len(final_patches_sequence):] = -1 # Use -1 to denote padded/ignored tokens

        return final_patches_sequence, is_noised_mask

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        self.x1 = int(x1); self.x2 = int(x2); self.y1 = int(y1); self.y2 = int(y2)
        assert self.x1<=self.x2 and self.y1<=self.y2
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
        
class FixedQuadTree:
    def __init__(self, domain, fixed_length=128) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()
    def _build_tree(self):
        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        self.nodes = [[root, root.contains(self.domain)]]
        while len(self.nodes) < self.fixed_length:
            if not self.nodes: break
            bbox, value = max(self.nodes, key=lambda x: x[1])
            idx = self.nodes.index([bbox, value])
            
            if bbox.get_size()[0] <= 2: break 
            x1,x2,y1,y2 = bbox.get_coord()
            cx, cy = (x1+x2)/2, (y1+y2)/2
            
            children = [
                Rect(x1, cx, cy, y2), Rect(cx, x2, cy, y2),
                Rect(x1, cx, y1, cy), Rect(cx, x2, y1, cy)
            ]
            children_with_values = [[child, child.contains(self.domain)] for child in children]
            
            self.nodes = self.nodes[:idx] + children_with_values + self.nodes[idx+1:]


if __name__ == '__main__':
    # 1. Create a dummy image and its edge map
    IMG_SIZE = 512
    original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
    original_image = cv2.circle(original_image, (256, 256), 120, (200, 150, 100), -1)
    original_image = cv2.circle(original_image, (180, 200), 30, (20, 40, 60), -1)
    
    # 2. Generate edge map
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)

    # 3. Build the Quadtree based on the edge map
    print("Building adaptive quadtree...")
    qdt = FixedQuadTree(domain=edges, fixed_length=512)
    print(f"Quadtree built with {len(qdt.nodes)} leaf nodes.")
    
    # 4. Use the HDEProcessor to create the training sequence
    print("Processing with HDE...")
    hde_processor = HDEProcessor(visible_fraction=0.25)
    final_sequence, noised_mask = hde_processor.create_training_sequence(original_image, qdt)
    
    print(f"Generated a sequence of {len(final_sequence)} patches.")
    print(f"Noised mask shape: {noised_mask.shape}")
    print(f"Number of visible patches: {np.sum(noised_mask == 0)}")
    print(f"Number of noised patches: {np.sum(noised_mask == 1)}")

    # 5. Visualize the result
    reconstructed_canvas = np.zeros_like(original_image)
    leaf_nodes_bboxes = [node for node, value in qdt.nodes]

    for i, patch in enumerate(final_sequence):
        if i >= len(leaf_nodes_bboxes): break 
        bbox = leaf_nodes_bboxes[i]
        
        patch_size = bbox.get_size()
        if patch_size[0] == 0 or patch_size[1] == 0: continue
        
        resized_patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_NEAREST)
        
        x1, x2, y1, y2 = bbox.get_coord()
        reconstructed_canvas[y1:y2, x1:x2, :] = resized_patch

    # Display using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Map (Guide)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(reconstructed_canvas, cv2.COLOR_BGR2RGB))
    axes[2].set_title('HDE Input (25% Visible, 75% Noised)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
