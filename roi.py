# roi.py
import cv2
import json
import numpy as np
import os


def create_resizable_window(win_name, max_w=1600, max_h=1000):
    """创建可调整大小的显示窗口"""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win_name, max_w, max_h)
    except Exception:
        pass


def resize_image_to_fit_window(img, max_w=1600, max_h=1000):
    """将图像缩放到适应窗口大小"""
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 0.999:
        return img.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def select_manual_roi_on_chess(chess_img_path, roi_cache_path=None):
    """
    在棋盘图上手动选择多边形ROI区域

    参数:
        chess_img_path: 棋盘图路径
        roi_cache_path: ROI缓存文件路径，如果存在则跳过选择直接加载

    返回:
        roi_points: 多边形点坐标列表 [[x1,y1], [x2,y2], ...]
    """
    # 优先尝试从缓存文件加载
    if roi_cache_path and os.path.exists(roi_cache_path):
        print(f"[ROI] 尝试从缓存加载: {roi_cache_path}")
        try:
            roi_data = load_roi_from_json(roi_cache_path)
            if roi_data and roi_data.get("roi_points"):
                print("[ROI] 从缓存成功加载ROI，跳过手动选择")
                return np.array(roi_data["roi_points"], dtype=np.int32)
        except Exception as e:
            print(f"[ROI] 缓存加载失败: {e}，将手动选择")

    # 读取棋盘图
    img = cv2.imread(chess_img_path)
    if img is None:
        raise RuntimeError(f"棋盘图读取失败: {chess_img_path}")

    points = []
    win_name = "Select ROI (Press 'q' to finish)"
    img_display = resize_image_to_fit_window(img.copy())

    def redraw():
        nonlocal img_display
        img_display = resize_image_to_fit_window(img.copy())
        if len(points) == 0:
            cv2.imshow(win_name, img_display)
            return

        scale_x = img.shape[1] / img_display.shape[1]
        scale_y = img.shape[0] / img_display.shape[0]
        display_pts = []
        for px, py in points:
            dx = int(round(px / scale_x))
            dy = int(round(py / scale_y))
            display_pts.append((dx, dy))
            cv2.circle(img_display, (dx, dy), 5, (0, 255, 0), -1)
        for i in range(1, len(display_pts)):
            cv2.line(img_display, display_pts[i - 1], display_pts[i], (255, 0, 0), 2)
        if len(display_pts) >= 3:
            cv2.polylines(img_display, [np.array(display_pts, dtype=np.int32)], True, (0, 255, 255), 2)
        cv2.imshow(win_name, img_display)

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale_x = img.shape[1] / img_display.shape[1]
        scale_y = img.shape[0] / img_display.shape[0]
        orig_x = int(round(x * scale_x))
        orig_y = int(round(y * scale_y))
        points.append([orig_x, orig_y])
        redraw()

    create_resizable_window(win_name)
    cv2.imshow(win_name, img_display)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("请点击鼠标左键选择多边形顶点（至少3个点）")
    print("按 'q' 键完成选择，按 'r' 键重新选择")
    print("完成后ROI将自动保存，下次处理同组图片时可直接使用")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if len(points) >= 3:
                break
            else:
                print("至少需要选择3个点")
        elif key == ord('r'):
            points = []
            redraw()
            print("已重置，请重新选择")

    cv2.destroyWindow(win_name)

    # 将点转换为numpy数组
    roi_points = np.array(points, dtype=np.int32)

    # 自动保存到缓存
    if roi_cache_path:
        try:
            roi_data = {
                "chessboard_image": os.path.basename(chess_img_path),
                "image_size": [img.shape[1], img.shape[0]],  # [width, height]
                "roi_points": roi_points.tolist(),
                "roi_type": "polygon"
            }
            save_roi_to_json(roi_data, roi_cache_path)
            print(f"[ROI] ROI已保存到: {roi_cache_path}")
        except Exception as e:
            print(f"[ROI] ROI保存失败: {e}")

    return roi_points


def roi_to_serializable(roi):
    """将ROI转换为可序列化格式（JSON友好）"""
    if roi is None:
        return None
    if isinstance(roi, np.ndarray):
        return roi.astype(int).tolist()
    return roi


def apply_roi_mask(img, roi):
    """
    应用ROI掩码到图像

    参数:
        img: 输入图像
        roi: ROI区域，可以是矩形(x,y,w,h)或多边形点列表

    返回:
        应用了ROI掩码的图像（ROI外部区域置为0）
    """
    if roi is None:
        return img.copy()

    out = np.zeros_like(img)
    if isinstance(roi, np.ndarray):
        # 多边形ROI
        poly = roi.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        if img.ndim == 2:
            out[mask > 0] = img[mask > 0]
        else:
            out[mask > 0] = img[mask > 0]
        return out
    elif isinstance(roi, (list, tuple)) and len(roi) == 4:
        # 矩形ROI
        x, y, w, h = roi
        out[y:y + h, x:x + w] = img[y:y + h, x:x + w]
        return out
    else:
        raise ValueError(f"不支持的ROI格式: {type(roi)}")


def draw_roi(vis, roi, color=(0, 255, 0), thickness=2):
    """
    在可视化图像上绘制ROI轮廓

    参数:
        vis: 原始图像
        roi: ROI区域
        color: 绘制颜色
        thickness: 线宽

    返回:
        绘制了ROI的图像
    """
    if roi is None:
        return vis

    if isinstance(roi, np.ndarray):
        # 多边形ROI
        poly = roi.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly], True, color, thickness)
        return vis
    elif isinstance(roi, (list, tuple)) and len(roi) == 4:
        # 矩形ROI
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        return vis
    else:
        return vis


def save_roi_to_json(roi_data, file_path):
    """
    保存ROI数据到JSON文件

    参数:
        roi_data: ROI数据字典，包含:
            - chessboard_image: 棋盘图文件名
            - image_size: 图像尺寸 [width, height]
            - roi_points: ROI点坐标列表
            - roi_type: ROI类型 ("polygon" 或 "rectangle")
        file_path: JSON文件保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2, ensure_ascii=False)
    return file_path


def load_roi_from_json(file_path):
    """
    从JSON文件加载ROI数据

    参数:
        file_path: JSON文件路径

    返回:
        roi_data: ROI数据字典，失败时返回None
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            roi_data = json.load(f)

        # 验证必要的字段
        required_fields = ["chessboard_image", "image_size", "roi_points"]
        for field in required_fields:
            if field not in roi_data:
                print(f"[ROI] JSON文件缺少必要字段: {field}")
                return None

        return roi_data
    except Exception as e:
        print(f"[ROI] 加载ROI JSON失败: {e}")
        return None


def get_roi_cache_path(chess_img_path, save_dir):
    """
    根据棋盘图路径生成ROI缓存文件路径

    参数:
        chess_img_path: 棋盘图路径
        save_dir: 结果保存目录

    返回:
        roi_cache_path: ROI缓存文件路径
    """
    chess_name = os.path.splitext(os.path.basename(chess_img_path))[0]
    cache_dir = os.path.join(save_dir, "roi_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{chess_name}_roi.json")


def load_or_select_roi(chess_img_path, save_dir, force_reselect=False):
    """
    智能ROI加载/选择函数

    参数:
        chess_img_path: 棋盘图路径
        save_dir: 结果保存目录
        force_reselect: 是否强制重新选择（忽略缓存）

    返回:
        roi_points: ROI点坐标数组
        roi_cache_path: 使用的缓存文件路径
    """
    roi_cache_path = get_roi_cache_path(chess_img_path, save_dir)

    if force_reselect:
        print("[ROI] 强制重新选择ROI")
        return select_manual_roi_on_chess(chess_img_path, roi_cache_path), roi_cache_path

    # 检查缓存是否存在
    if os.path.exists(roi_cache_path):
        roi_data = load_roi_from_json(roi_cache_path)
        if roi_data and roi_data.get("roi_points"):
            print(f"[ROI] 从缓存加载ROI: {roi_cache_path}")
            return np.array(roi_data["roi_points"], dtype=np.int32), roi_cache_path

    # 缓存不存在或加载失败，手动选择
    print(f"[ROI] 未找到有效缓存，开始手动选择ROI")
    return select_manual_roi_on_chess(chess_img_path, roi_cache_path), roi_cache_path


# roi.py 新增函数
def derive_group_info_from_chess(chess_path):
    """
    从棋盘图文件名提取组信息

    参数:
        chess_path: 棋盘图文件路径

    返回:
        tuple: (prefix, chess_idx, light_pattern)
            - prefix: 文件名前缀（如 undist_d2p0_y-0p1_p-59p7）
            - chess_idx: 棋盘索引（如 001）
            - light_pattern: 匹配的光源图模式
    """
    import re
    import os
    import glob

    name = os.path.basename(chess_path)
    m = re.match(r"^(.*)_chess_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", name, re.IGNORECASE)
    if not m:
        raise ValueError("棋盘格文件名不符合命名规则，应类似：undist_d1p0_y0p0_p-7p2_chess_001.jpg")
    prefix = m.group(1)
    chess_idx = m.group(2)
    ext = m.group(3)
    folder = os.path.dirname(chess_path)
    light_pattern = os.path.join(folder, f"{prefix}_light_{chess_idx}_exp_*.{ext}")
    return prefix, chess_idx, light_pattern


def find_light_images_by_chess(chess_path):
    """
    根据棋盘图查找匹配的光源图

    参数:
        chess_path: 棋盘图文件路径

    返回:
        list: 排序后的光源图文件路径列表
    """
    _, _, light_pattern = derive_group_info_from_chess(chess_path)
    return sorted(glob.glob(light_pattern))
