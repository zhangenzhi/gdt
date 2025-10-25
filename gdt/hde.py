import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
import argparse

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
        This version now also tracks the depth of each leaf node.
        """
        h, w, c = img.shape
        
        # Initialize the canvas that will hold the final accumulated noise
        noise_canvas = np.zeros((h, w, c), dtype=np.float32)
        
        # Initialize the tree building process
        root = Rect(0, w, 0, h)
        
        # The list of active leaf nodes, storing [priority_score, Rect_object, depth]
        # The root node has a depth of 0.
        active_nodes = [[root.contains(edge_map), root, 0]]

        # The final list of leaf nodes, storing [Rect_object, depth]
        final_leaf_nodes_with_depth = []

        # --- Base Noise (Level 0) ---
        # Calculate and apply noise for the entire image
        root_patch = root.get_area(img)
        target_variance = np.mean(root_patch.astype(np.float32))
        if target_variance > 0:
            std_dev = np.sqrt(target_variance)
            noise = np.random.normal(loc=0.0, scale=std_dev, size=root_patch.shape)
            noise_canvas += noise
        
        # --- Iterative Splitting and Noise Addition ---
        while len(active_nodes) + len(final_leaf_nodes_with_depth) < fixed_length:
            if not active_nodes: break
            
            # Find the leaf with the highest edge score to split
            score, bbox_to_split, depth = max(active_nodes, key=lambda x: x[0])
            
            # If the patch is too small or has no edges, don't split it further.
            # Move it to the final list.
            if bbox_to_split.get_size()[0] <= 2 or score == 0:
                final_leaf_nodes_with_depth.append([bbox_to_split, depth])
                active_nodes.remove([score, bbox_to_split, depth])
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
                x_c1, x_c2, y_c1, y_c2 = child_rect.get_coord()
                existing_noise_patch = noise_canvas[y_c1:y_c2, x_c1:x_c2, :]
                current_variance = np.var(existing_noise_patch)
                img_patch = child_rect.get_area(img)
                target_variance = np.mean(img_patch.astype(np.float32))
                new_variance = max(0, target_variance - current_variance)
                
                if new_variance > 0:
                    new_std_dev = np.sqrt(new_variance)
                    new_noise = np.random.normal(loc=0.0, scale=new_std_dev, size=img_patch.shape)
                    noise_canvas[y_c1:y_c2, x_c1:x_c2, :] += new_noise

                # Add the new child with incremented depth to the active list
                child_score = child_rect.contains(edge_map)
                active_nodes.append([child_score, child_rect, depth + 1])

            # Remove the parent from the active list
            active_nodes.remove([score, bbox_to_split, depth])

        # Combine the remaining active nodes with the finalized ones
        all_leaf_nodes_with_depth = final_leaf_nodes_with_depth + [[node, d] for _, node, d in active_nodes]
        
        # Sort nodes by top-to-bottom, left-to-right for a consistent spatial order
        all_leaf_nodes_with_depth.sort(key=lambda item: (item[0].y1, item[0].x1))
        
        # Unzip into separate lists
        all_leaf_nodes = [item[0] for item in all_leaf_nodes_with_depth]
        all_leaf_depths = [item[1] for item in all_leaf_nodes_with_depth]

        return noise_canvas, all_leaf_nodes, all_leaf_depths

    def create_training_sequence(self, img, edge_map, fixed_length):
        """
        Creates the final training sequence using the hierarchical noise method.
        """
        # 1. Generate the hierarchical noise, leaves, and their depths
        noise_canvas, leaf_nodes, leaf_depths = self.generate_hierarchical_noise_canvas(img, edge_map, fixed_length)
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
                x1, x2, y1, y2 = bbox.get_coord()
                hierarchical_noise = noise_canvas[y1:y2, x1:x2, :]
                noised_patch_float = original_patch.astype(np.float32) + hierarchical_noise
                noised_patch = np.clip(noised_patch_float, 0, 255).astype(np.uint8)
                final_patches_sequence.append(noised_patch)

        # Handle padding for the mask if fewer than fixed_length patches were generated
        if len(leaf_nodes) < fixed_length:
            is_noised_mask[len(leaf_nodes):] = -1

        return final_patches_sequence, leaf_nodes, is_noised_mask, noise_canvas, leaf_depths

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
        
# --- Example Usage ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical HDE Visualization Demo')
    # parser.add_argument('--image_path', type=str, default="/Users/zezzz/Desktop/zez/Me_and_My_Research/hde/rescaled_image_0_4096x4096.png", help='Path to a local image to process.')
    parser.add_argument('--image_path', type=str, default="/Users/zezzz/Desktop/zez/Me_and_My_Research/hde/hydrogel_16_1024x1024.jpg", help='Path to a local image to process.')
    # parser.add_argument('--image_path', type=str, default="/Users/zezzz/Desktop/zez/Me_and_My_Research/hde/rescaled_image_0_4096x4096.png", help='Path to a local image to process.')
    args = parser.parse_args()
            
    # --- Start of Demo ---
    IMG_SIZE = 512
    FIXED_LENGTH = 256
    
    # Load image from path or create a synthetic one
    if args.image_path:
        original_image = cv2.imread(args.image_path)
        if original_image is None:
            print(f"Error: Could not load image from {args.image_path}. Using synthetic image.")
            # Fallback to synthetic image
            original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
            original_image = cv2.circle(original_image, (128, 128), 90, (200, 150, 100), -1)
        else:
            print(f"Loaded image from {args.image_path}")
            original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
    else:
        print("No image path provided. Using synthetic image.")
        original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
        original_image = cv2.circle(original_image, (128, 128), 90, (200, 150, 100), -1)
        original_image = cv2.circle(original_image, (90, 100), 20, (20, 40, 60), -1)
    
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)

    print("Processing with NEW HierarchicalHDEProcessor...")
    hierarchical_processor = HierarchicalHDEProcessor(visible_fraction=0.25)
    final_sequence, leaf_nodes, noised_mask, noise_canvas, leaf_depths = hierarchical_processor.create_training_sequence(
        original_image, edges, fixed_length=FIXED_LENGTH
    )
    
    print(f"Generated a sequence of {len(final_sequence)} patches.")
    print(f"Number of visible patches: {np.sum(noised_mask == 0)}")
    print(f"Number of noised patches: {np.sum(noised_mask == 1)}")

    # --- Visualize the result ---
    # Create the canvas for the HDE input visualization
    hde_input_canvas = np.zeros_like(original_image, dtype=np.float32)

    for i, bbox in enumerate(leaf_nodes):
        if i >= len(noised_mask): break
        x1, x2, y1, y2 = bbox.get_coord()
        patch_size = bbox.get_size()
        if patch_size[0] == 0 or patch_size[1] == 0: continue

        is_noised = noised_mask[i] == 1
        if is_noised:
            noise_patch = noise_canvas[y1:y2, x1:x2, :]
            noise_vis = noise_patch - np.mean(noise_patch)
            if np.std(noise_vis) > 1e-6:
                noise_vis = noise_vis / (np.std(noise_vis) * 3)
            noise_vis = (noise_vis + 0.5).clip(0, 1)
            hde_input_canvas[y1:y2, x1:x2, :] = noise_vis
        else:
            original_patch = bbox.get_area(original_image)
            hde_input_canvas[y1:y2, x1:x2, :] = original_patch.astype(np.float32) / 255.0

    if hde_input_canvas.shape[2] == 3:
        hde_input_canvas_rgb = cv2.cvtColor((hde_input_canvas * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
    else:
        hde_input_canvas_rgb = hde_input_canvas


    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Panel 1: 原始图像 (Original Image)
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Panel 2: 边缘图与自适应网格 (Edge Map + Quadtree Grid)
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Map + Quadtree')
    for bbox in leaf_nodes:
        rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
                             linewidth=0.8, edgecolor='cyan', facecolor='none')
        axes[0, 1].add_patch(rect)
    axes[0, 1].axis('off')
    
    # Panel 3: HDE 模型输入 (HDE Input)
    axes[1, 0].imshow(hde_input_canvas_rgb)
    axes[1, 0].set_title('DE Input: Clean=G, Noised=R')
    for i, bbox in enumerate(leaf_nodes):
         if i >= len(noised_mask): break
         is_noised = noised_mask[i] == 1
         edge_color = 'r' if is_noised else 'g'
         rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
                              linewidth=0.8, edgecolor=edge_color, facecolor='none')
         axes[1, 0].add_patch(rect)
    axes[1, 0].axis('off')
    
    # Panel 4: 噪声结构图 (Patch-level Noise Variance)
    patch_variance_canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, bbox in enumerate(leaf_nodes):
        x1, x2, y1, y2 = bbox.get_coord()
        
        # Extract the noise from the corresponding patch
        noise_patch = noise_canvas[y1:y2, x1:x2, :]
        
        # Calculate the variance within that patch
        patch_variance = np.var(noise_patch)
        
        # Fill the canvas with this variance value
        patch_variance_canvas[y1:y2, x1:x2] = patch_variance

    # Normalize the entire canvas for visualization
    min_var, max_var = np.min(patch_variance_canvas), np.max(patch_variance_canvas)
    if max_var > min_var:
        patch_variance_canvas_vis = (patch_variance_canvas - min_var) / (max_var - min_var)
    else:
        patch_variance_canvas_vis = np.zeros_like(patch_variance_canvas)
    
    im = axes[1, 1].imshow(patch_variance_canvas_vis, cmap='inferno')
    axes[1, 1].set_title('Noise Structure')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
