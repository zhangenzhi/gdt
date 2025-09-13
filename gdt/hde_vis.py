import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import imageio
import os

# NOTE: This is a self-contained script for generating a GIF visualization.
# It re-uses the core logic from your hde_utils.py file.

# --- Helper Classes & Functions ---

class Rect:
    """A helper class to manage rectangular patches."""
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

def generate_frame(step, original_image, edges, leaf_nodes, noise_canvas):
    """Generates a single frame (as a NumPy array) for the GIF."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"HDE Hierarchical Noise Generation | Step: {step}", fontsize=16)

    # Panel 1: Original Image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像 (Original Image)')
    axes[0, 0].axis('off')

    # Panel 2: Edge Map + Evolving Quadtree Grid
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('边缘图与演化中的网格 (Edge Map + Grid)')
    for bbox in leaf_nodes:
        rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
                             linewidth=0.8, edgecolor='cyan', facecolor='none')
        axes[0, 1].add_patch(rect)
    axes[0, 1].axis('off')
    
    # Panel 3: HDE Input (Conceptual)
    axes[1, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('概念输入 (Conceptual Input)')
    axes[1, 0].axis('off')
    for bbox in leaf_nodes:
        rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
                             linewidth=0.8, edgecolor='gray', alpha=0.5, facecolor='none')
        axes[1, 0].add_patch(rect)
        
    # Panel 4: Hierarchical Noise Structure (Patch-level)
    h, w, _ = original_image.shape
    patch_variance_canvas = np.zeros((h, w), dtype=np.float32)
    for bbox in leaf_nodes:
        x1, x2, y1, y2 = bbox.get_coord()
        noise_patch = noise_canvas[y1:y2, x1:x2, :]
        patch_variance = np.var(noise_patch)
        patch_variance_canvas[y1:y2, x1:x2] = patch_variance
    
    # Normalize the entire canvas for visualization
    min_var, max_var = np.min(patch_variance_canvas), np.max(patch_variance_canvas)
    # Use a small epsilon to avoid division by zero if max_var == min_var
    if max_var > min_var + 1e-6:
        patch_variance_canvas_vis = (patch_variance_canvas - min_var) / (max_var - min_var)
    else:
        patch_variance_canvas_vis = np.zeros_like(patch_variance_canvas)
        
    im = axes[1, 1].imshow(patch_variance_canvas_vis, cmap='inferno', vmin=0, vmax=1)
    axes[1, 1].set_title('噪声结构图 (Patch-level Variance)')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # --- FIXED: Robust method for converting plot to numpy array ---
    fig.canvas.draw()
    # Get pixel dimensions from figure size and DPI
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
    # Get the RGBA buffer from the canvas
    buf = fig.canvas.buffer_rgba()
    # Convert the buffer to a NumPy array and reshape
    frame_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    # Convert from RGBA to RGB by dropping the alpha channel
    frame_rgb = frame_rgba[:, :, :3]
    plt.close(fig)
    return frame_rgb


def create_process_visualization(img, edge_map, fixed_length, output_path):
    """
    Simulates the HDE process step-by-step and generates a GIF.
    """
    h, w, c = img.shape
    noise_canvas = np.zeros((h, w, c), dtype=np.float32)
    root = Rect(0, w, 0, h)
    
    # active_nodes: [priority_score, Rect_object, depth]
    active_nodes = [[root.contains(edge_map), root, 0]]
    final_leaf_nodes_with_depth = []
    
    frames = []
    step = 0
    
    # Initial frame (before any splits)
    print("Generating frame 0...")
    frames.append(generate_frame(step, img, edge_map, [root], noise_canvas))
    
    # --- Base Noise (Level 0) ---
    root_patch = root.get_area(img)
    target_variance = np.mean(root_patch.astype(np.float32))
    if target_variance > 0:
        noise = np.random.normal(0, np.sqrt(target_variance), root_patch.shape)
        noise_canvas += noise

    # --- Iterative Splitting and Frame Generation ---
    while len(active_nodes) + len(final_leaf_nodes_with_depth) < fixed_length:
        if not active_nodes: break
        step += 1
        
        score, bbox_to_split, depth = max(active_nodes, key=lambda x: x[0])
        
        if bbox_to_split.get_size()[0] <= 2 or score == 0:
            final_leaf_nodes_with_depth.append([bbox_to_split, depth])
            active_nodes.remove([score, bbox_to_split, depth])
            continue
            
        # Perform the split
        x1, x2, y1, y2 = bbox_to_split.get_coord()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        children_rects = [
            Rect(x1, cx, y1, cy), Rect(cx, x2, y1, cy),
            Rect(x1, cx, cy, y2), Rect(cx, x2, cy, y2)
        ]
        
        for child_rect in children_rects:
            x_c1, x_c2, y_c1, y_c2 = child_rect.get_coord()
            existing_noise_patch = noise_canvas[y_c1:y_c2, x_c1:x_c2, :]
            current_variance = np.var(existing_noise_patch)
            img_patch = child_rect.get_area(img)
            target_variance = np.mean(img_patch.astype(np.float32))
            new_variance = max(0, target_variance - current_variance)
            
            if new_variance > 0:
                new_noise = np.random.normal(0, np.sqrt(new_variance), img_patch.shape)
                noise_canvas[y_c1:y_c2, x_c1:x_c2, :] += new_noise

            child_score = child_rect.contains(edge_map)
            active_nodes.append([child_score, child_rect, depth + 1])
        
        active_nodes.remove([score, bbox_to_split, depth])
        
        # Generate a frame every few steps to keep the GIF concise
        if step % 5 == 0 or len(active_nodes) + len(final_leaf_nodes_with_depth) >= fixed_length:
            print(f"Generating frame at step {step}...")
            current_leaves = [node for _, node, _ in active_nodes] + [node for node, _ in final_leaf_nodes_with_depth]
            frames.append(generate_frame(step, img, edge_map, current_leaves, noise_canvas))

    # --- Save the GIF ---
    print(f"Saving animation to {output_path}...")
    # FIXED: Replaced deprecated `fps` with `duration`. duration is in ms.
    # 5 fps = 1000ms / 5 = 200ms per frame.
    imageio.mimsave(output_path, frames, duration=200)
    print("Done.")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical HDE Process Visualization Tool')
    parser.add_argument('--image_path', type=str, default="/Users/zezzz/Desktop/zez/Me_and_My_Research/hde/rescaled_image_0_4096x4096.png", help='Path to a local image to process.')
    parser.add_argument('--output_path', type=str, default="hde_process_visualization.gif", help='Path to save the output GIF.')
    parser.add_argument('--img_size', type=int, default=512, help='Size to resize the input image to.')
    parser.add_argument('--patches', type=int, default=256, help='Number of final quadtree patches.')
    args = parser.parse_args()

    # Load and prepare the image
    original_image = cv2.imread(args.image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image from {args.image_path}")
    
    original_image = cv2.resize(original_image, (args.img_size, args.img_size))
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)
    
    # Create the visualization
    create_process_visualization(original_image, edges, args.patches, args.output_path)

