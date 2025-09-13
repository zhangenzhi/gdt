import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

# We assume the Rect and FixedQuadTree classes from your provided code are available.
# This file can be imported into your main script.

class HDEProcessor:
    # ... (The original, simple HDEProcessor class is kept for reference or comparison)
    """
    (Original Independent Noise Processor)
    Encapsulates the logic for the HDE pre-training task with independent noise generation.
    """
    def __init__(self, visible_fraction=0.25):
        assert 0.0 < visible_fraction < 1.0
        self.visible_fraction = visible_fraction

    def generate_noise_for_patch(self, patch):
        patch_float = patch.astype(np.float32)
        avg_p = np.mean(patch_float)
        if avg_p <= 0:
            return np.zeros_like(patch_float)
        std_dev = np.sqrt(avg_p)
        return np.random.normal(loc=0.0, scale=std_dev, size=patch.shape)

    def create_training_sequence(self, img, qdt):
        leaf_nodes = [node for node, value in qdt.nodes]
        num_patches = len(leaf_nodes)
        indices = list(range(num_patches))
        random.shuffle(indices)
        num_visible = int(num_patches * self.visible_fraction)
        visible_indices = set(indices[:num_visible])
        
        final_patches_sequence = []
        is_noised_mask = np.zeros(qdt.fixed_length, dtype=int)

        for i, bbox in enumerate(leaf_nodes):
            original_patch = bbox.get_area(img)
            if i in visible_indices:
                final_patches_sequence.append(original_patch)
                is_noised_mask[i] = 0
            else:
                is_noised_mask[i] = 1
                noise = self.generate_noise_for_patch(original_patch)
                noised_patch_float = original_patch.astype(np.float32) + noise
                noised_patch = np.clip(noised_patch_float, 0, 255).astype(np.uint8)
                final_patches_sequence.append(noised_patch)
        
        padding_needed = qdt.fixed_length - len(final_patches_sequence)
        if padding_needed > 0:
            is_noised_mask[len(final_patches_sequence):] = -1

        return final_patches_sequence, is_noised_mask


class HierarchicalHDEProcessor:
    """
    (New Hierarchical Noise Processor)
    Implements the advanced HDE pre-training task with hierarchical, cumulative noise.
    Noise is generated and accumulated during the tree-building process,
    and includes variance suppression to prevent noise explosion.
    """
    def __init__(self, visible_fraction=0.25):
        """
        Initializes the Hierarchical HDE processor.
        """
        assert 0.0 < visible_fraction < 1.0, "visible_fraction must be between 0 and 1."
        self.visible_fraction = visible_fraction

    def generate_hierarchical_noise_canvas(self, img, edge_map, fixed_length):
        """
        Builds the quadtree and generates a hierarchical noise canvas simultaneously.
        """
        h, w, c = img.shape
        
        # Initialize the canvas that will hold the final accumulated noise
        noise_canvas = np.zeros((h, w, c), dtype=np.float32)
        
        # Initialize the tree building process
        root = Rect(0, w, 0, h)
        
        # The list of active leaf nodes, storing (priority_score, Rect_object)
        # We use edge count as the priority score.
        active_nodes = [[root.contains(edge_map), root]]

        # The final list of leaf nodes in their spatial order
        final_leaf_nodes = []

        # --- Base Noise (Level 0) ---
        # Calculate and apply noise for the entire image
        root_patch = root.get_area(img)
        target_variance = np.mean(root_patch.astype(np.float32))
        if target_variance > 0:
            std_dev = np.sqrt(target_variance)
            noise = np.random.normal(loc=0.0, scale=std_dev, size=root_patch.shape)
            noise_canvas += noise
        
        # --- Iterative Splitting and Noise Addition ---
        while len(active_nodes) + len(final_leaf_nodes) < fixed_length:
            if not active_nodes: break
            
            # Find the leaf with the highest edge score to split
            score, bbox_to_split = max(active_nodes, key=lambda x: x[0])
            
            # If the patch is too small or has no edges, don't split it further.
            # Move it to the final list.
            if bbox_to_split.get_size()[0] <= 2 or score == 0:
                final_leaf_nodes.append(bbox_to_split)
                active_nodes.remove([score, bbox_to_split])
                continue

            # Split the chosen bbox into 4 children
            x1, x2, y1, y2 = bbox_to_split.get_coord()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            children_rects = [
                Rect(x1, cx, y1, cy), Rect(cx, x2, y1, cy),
                Rect(x1, cx, cy, y2), Rect(cx, x2, cy, y2)
            ]
            
            # For each new child, add a new layer of noise
            for child_rect in children_rects:
                # --- Variance Suppression Logic ---
                # 1. Get the noise already present in this child's region
                x_c1, x_c2, y_c1, y_c2 = child_rect.get_coord()
                existing_noise_patch = noise_canvas[y_c1:y_c2, x_c1:x_c2, :]
                current_variance = np.var(existing_noise_patch)
                
                # 2. Get the target variance from the image content
                img_patch = child_rect.get_area(img)
                target_variance = np.mean(img_patch.astype(np.float32))
                
                # 3. Calculate the variance for the *new* noise layer
                new_variance = max(0, target_variance - current_variance)
                
                if new_variance > 0:
                    new_std_dev = np.sqrt(new_variance)
                    new_noise = np.random.normal(loc=0.0, scale=new_std_dev, size=img_patch.shape)
                    
                    # 4. Add the new noise layer to the canvas
                    noise_canvas[y_c1:y_c2, x_c1:x_c2, :] += new_noise

                # Add the new child to the list of active nodes for potential future splits
                child_score = child_rect.contains(edge_map)
                active_nodes.append([child_score, child_rect])

            # Remove the parent from the active list
            active_nodes.remove([score, bbox_to_split])

        # Combine the remaining active nodes with the finalized ones
        all_leaf_nodes = final_leaf_nodes + [node for _, node in active_nodes]
        
        # Sort nodes by top-to-bottom, left-to-right for a consistent spatial order
        all_leaf_nodes.sort(key=lambda r: (r.y1, r.x1))

        return noise_canvas, all_leaf_nodes

    def create_training_sequence(self, img, edge_map, fixed_length):
        """
        Creates the final training sequence using the hierarchical noise method.
        """
        # 1. Generate the hierarchical noise and the corresponding quadtree leaves
        noise_canvas, leaf_nodes = self.generate_hierarchical_noise_canvas(img, edge_map, fixed_length)
        num_patches = len(leaf_nodes)
        
        # 2. Randomly select which patches will be visible
        indices = list(range(num_patches))
        random.shuffle(indices)
        num_visible = int(num_patches * self.visible_fraction)
        visible_indices = set(indices[:num_visible])
        
        final_patches_sequence = []
        is_noised_mask = np.zeros(fixed_length, dtype=int)

        # 3. Create the final sequence of clean and noised patches
        for i, bbox in enumerate(leaf_nodes):
            original_patch = bbox.get_area(img)
            
            if i in visible_indices:
                final_patches_sequence.append(original_patch)
                is_noised_mask[i] = 0  # 0 for visible
            else:
                is_noised_mask[i] = 1  # 1 for noised
                
                # Get the hierarchical noise from the canvas
                x1, x2, y1, y2 = bbox.get_coord()
                hierarchical_noise = noise_canvas[y1:y2, x1:x2, :]
                
                noised_patch_float = original_patch.astype(np.float32) + hierarchical_noise
                
                # Clip to valid range
                noised_patch = np.clip(noised_patch_float, 0, 255).astype(np.uint8)
                final_patches_sequence.append(noised_patch)

        # Handle padding for the mask if fewer than fixed_length patches were generated
        padding_needed = fixed_length - num_patches
        if padding_needed > 0:
            is_noised_mask[num_patches:] = -1 # Use -1 for padded/ignored tokens

        return final_patches_sequence, leaf_nodes, is_noised_mask

# --- Example Usage ---
if __name__ == '__main__':
    # For this self-contained example, we include the necessary base classes.
    class Rect:
        def __init__(self, x1, x2, y1, y2) -> None:
            self.x1 = int(x1); self.x2 = int(x2); self.y1 = int(y1); self.y2 = int(y2)
            assert self.x1 <= self.x2 and self.y1 <= self.y2
        def contains(self, domain):
            if self.y1 >= self.y2 or self.x1 >= self.x2: return 0
            patch = domain[self.y1:self.y2, self.x1:self.x2]
            return int(np.sum(patch) / 255)
        def get_area(self, img):
            return img[self.y1:self.y2, self.x1:self.x2, :]
        def get_coord(self):
            return self.x1, self.x2, self.y1, self.y2
        def get_size(self):
            return self.x2 - self.x1, self.y2 - self.y1
            
    # --- Start of Demo ---
    IMG_SIZE = 256
    FIXED_LENGTH = 196
    
    original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
    original_image = cv2.circle(original_image, (128, 128), 90, (200, 150, 100), -1)
    original_image = cv2.circle(original_image, (90, 100), 20, (20, 40, 60), -1)
    
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)

    print("Processing with NEW HierarchicalHDEProcessor...")
    hierarchical_processor = HierarchicalHDEProcessor(visible_fraction=0.25)
    final_sequence, leaf_nodes, noised_mask = hierarchical_processor.create_training_sequence(
        original_image, edges, fixed_length=FIXED_LENGTH
    )
    
    print(f"Generated a sequence of {len(final_sequence)} patches.")
    print(f"Number of visible patches: {np.sum(noised_mask == 0)}")
    print(f"Number of noised patches: {np.sum(noised_mask == 1)}")

    # Visualize the result
    reconstructed_canvas = np.zeros_like(original_image)
    for i, patch in enumerate(final_sequence):
        if i >= len(leaf_nodes): break
        bbox = leaf_nodes[i]
        patch_size = bbox.get_size()
        if patch_size[0] == 0 or patch_size[1] == 0: continue
        resized_patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_NEAREST)
        x1, x2, y1, y2 = bbox.get_coord()
        reconstructed_canvas[y1:y2, x1:x2, :] = resized_patch

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Map (Guide)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(reconstructed_canvas, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Hierarchical HDE Input (25% Visible)')
    for i, bbox in enumerate(leaf_nodes):
         if i >= len(noised_mask): break
         is_noised = noised_mask[i] == 1
         edge_color = 'r' if is_noised else 'g'
         rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
                              linewidth=0.8, edgecolor=edge_color, facecolor='none')
         axes[2].add_patch(rect)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

