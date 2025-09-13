import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# --- 配置参数 (Configuration Parameters) ---
# 您可以在这里调整图表的视觉风格
BASE_COLOR = '#4a4a4a'       # 基础形状颜色
GRID_COLOR = '#00ffff'       # 网格线颜色 (青色)
# 为不同层级的噪声定义一个颜色列表 (增加了一种颜色以支持更深的层级)
NOISE_COLORS = ['#ffeb3b', '#ff9800', '#f44336', '#9C27B0'] # 黄 -> 橙 -> 红 -> 紫
ARROW_COLOR = '#ffffff'      # 箭头颜色
TEXT_COLOR = '#ffffff'       # 文本颜色
BG_COLOR = '#212121'         # 背景颜色

def get_perspective_polygon(center_x, center_y, width=300, height=120, skew=120):
    """根据中心点和参数计算水平倾斜的平行四边形顶点"""
    half_w, half_h = width / 2, height / 2
    # p1--p2
    # |  |
    # p4--p3
    p1 = (center_x - half_w - skew / 2, center_y + half_h)
    p2 = (center_x + half_w - skew / 2, center_y + half_h)
    p3 = (center_x + half_w + skew / 2, center_y - half_h)
    p4 = (center_x - half_w + skew / 2, center_y - half_h)
    return np.array([p1, p2, p3, p4])

def transform_point(p, poly):
    """将一个点 p 从单位正方形 [0,1]x[0,1] 映射到目标四边形内"""
    u, v = p
    p1, p2, p3, p4 = poly # p1=左上, p2=右上, p3=右下, p4=左下
    # 沿左右两边进行线性插值
    left_edge_pt = (1 - v) * p1 + v * p4
    right_edge_pt = (1 - v) * p2 + v * p3
    # 在左右两边的插值点之间再次进行线性插值
    return (1 - u) * left_edge_pt + u * right_edge_pt

def get_sub_polygons(parent_poly):
    """将一个倾斜四边形分割成四个子四边形"""
    p1, p2, p3, p4 = parent_poly
    
    top_mid = (p1 + p2) / 2
    bottom_mid = (p3 + p4) / 2
    left_mid = transform_point((0, 0.5), parent_poly)
    right_mid = transform_point((1, 0.5), parent_poly)
    center = transform_point((0.5, 0.5), parent_poly)
    
    poly1 = np.array([p1, top_mid, center, left_mid])     # 左上
    poly2 = np.array([top_mid, p2, right_mid, center])    # 右上
    poly3 = np.array([center, right_mid, p3, bottom_mid]) # 右下
    poly4 = np.array([left_mid, center, bottom_mid, p4])  # 左下
    
    return [poly1, poly2, poly3, poly4]

def draw_grid_on_polygon(ax, poly, level=1, max_level=1):
    """在倾斜四边形上递归绘制网格"""
    if level > max_level:
        return
        
    center = transform_point((0.5, 0.5), poly)
    left_mid = transform_point((0, 0.5), poly)
    right_mid = transform_point((1, 0.5), poly)
    top_mid = (poly[0] + poly[1]) / 2
    bottom_mid = (poly[2] + poly[3]) / 2
    
    # 绘制横向和纵向分割线
    ax.add_patch(patches.PathPatch(Path([left_mid, center, right_mid]), color=GRID_COLOR, lw=1.5, fill=False))
    ax.add_patch(patches.PathPatch(Path([top_mid, center, bottom_mid]), color=GRID_COLOR, lw=1.5, fill=False))


    if level < max_level:
        sub_polys = get_sub_polygons(poly)
        # 模拟自适应性：只在左上角的子块中递归细分
        draw_grid_on_polygon(ax, sub_polys[0], level + 1, max_level)
        # 在其他块中也进行一次细分以增加视觉密度
        draw_grid_on_polygon(ax, sub_polys[1], level + 1, max_level-1)


def draw_noise_on_polygon(ax, poly, level=1, max_level=1):
    """在倾斜四边形中递归绘制多层、不同颜色的示意性噪声点"""
    if level > max_level:
        return

    color = NOISE_COLORS[min(level - 1, len(NOISE_COLORS) - 1)]
    num_points = 40  # 每层噪声点的数量
    
    # 在单位正方形中生成随机点
    uv_points = np.random.rand(num_points, 2)
    # 将这些点转换到倾斜的四边形内部
    xy_points = np.array([transform_point(p, poly) for p in uv_points])
    
    ax.scatter(xy_points[:, 0], xy_points[:, 1], s=25/(level*0.8), color=color, alpha=0.8, edgecolors='none')

    if level < max_level:
        sub_polys = get_sub_polygons(poly)
        # 模拟自适应性：只在左上角和右上角的子块中递归添加更精细的噪声
        draw_noise_on_polygon(ax, sub_polys[0], level + 1, max_level)
        draw_noise_on_polygon(ax, sub_polys[1], level + 1, max_level-1)

def create_diagram(output_path="hde_conceptual_diagram.png"):
    """主函数，创建并保存示意图"""
    fig, ax = plt.subplots(figsize=(12, 18))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- 定义更紧凑的垂直布局 ---
    x_col_img = 350
    x_col_noise = 750
    y_steps = [1250, 800, 350]
    
    # --- 添加标题和列标签 ---
    fig.suptitle("HDE Noise Weaving Process", color=TEXT_COLOR, fontsize=28, weight='bold', y=0.92)
    ax.text(x_col_img, y_steps[0] + 180, "Image", color=TEXT_COLOR, ha='center', fontsize=20)
    ax.text(x_col_noise, y_steps[0] + 180, "Noise Canvas", color=TEXT_COLOR, ha='center', fontsize=20)
    
    # --- 阶段 k (起始阶段) ---
    y0 = y_steps[0]
    poly_img_0 = get_perspective_polygon(x_col_img, y0)
    poly_noise_0 = get_perspective_polygon(x_col_noise, y0)
    ax.add_patch(patches.Polygon(poly_img_0, closed=True, color=BASE_COLOR, alpha=0.5))
    ax.add_patch(patches.Polygon(poly_noise_0, closed=True, color=BASE_COLOR, alpha=0.5))
    draw_grid_on_polygon(ax, poly_img_0, max_level=1)
    draw_noise_on_polygon(ax, poly_noise_0, max_level=1)
    ax.text(550, y0 - 150, "Step k: Initial Grid & Noise", color=TEXT_COLOR, ha='center', fontsize=18)
    
    # --- 阶段 k+1 ---
    y1 = y_steps[1]
    poly_img_1 = get_perspective_polygon(x_col_img, y1)
    poly_noise_1 = get_perspective_polygon(x_col_noise, y1)
    ax.add_patch(patches.Polygon(poly_img_1, closed=True, color=BASE_COLOR, alpha=0.5))
    ax.add_patch(patches.Polygon(poly_noise_1, closed=True, color=BASE_COLOR, alpha=0.5))
    draw_grid_on_polygon(ax, poly_img_1, max_level=2)
    draw_noise_on_polygon(ax, poly_noise_1, max_level=1) 
    draw_noise_on_polygon(ax, poly_noise_1, max_level=2)
    ax.text(550, y1 - 150, "Step k+1: First Refinement", color=TEXT_COLOR, ha='center', fontsize=18)

    # --- 阶段 k+2 ---
    y2 = y_steps[2]
    poly_img_2 = get_perspective_polygon(x_col_img, y2)
    poly_noise_2 = get_perspective_polygon(x_col_noise, y2)
    ax.add_patch(patches.Polygon(poly_img_2, closed=True, color=BASE_COLOR, alpha=0.5))
    ax.add_patch(patches.Polygon(poly_noise_2, closed=True, color=BASE_COLOR, alpha=0.5))
    draw_grid_on_polygon(ax, poly_img_2, max_level=3)
    draw_noise_on_polygon(ax, poly_noise_2, max_level=1) 
    draw_noise_on_polygon(ax, poly_noise_2, max_level=2)
    draw_noise_on_polygon(ax, poly_noise_2, max_level=3)
    ax.text(550, y2 - 150, "Step k+2: Further Refinement", color=TEXT_COLOR, ha='center', fontsize=18)
    
    # # --- 添加箭头 ---
    # arrow_style = dict(arrowstyle="->", color=ARROW_COLOR, lw=3)
    # ax.add_patch(patches.FancyArrowPatch((550, y_steps[0] - 190), (550, y_steps[1] + 150), connectionstyle="arc3,rad=0", **arrow_style))
    # ax.add_patch(patches.FancyArrowPatch((550, y_steps[1] - 190), (550, y_steps[2] + 150), connectionstyle="arc3,rad=0", **arrow_style))

    # --- 设置画布 ---
    ax.set_xlim(0, 1100)
    ax.set_ylim(100, 1500)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor=BG_COLOR)
    print(f"示意图已保存至: {output_path}")

if __name__ == '__main__':
    create_diagram()

