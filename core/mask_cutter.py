import numpy as np
import cv2

def polygons_to_mask(shape, polygons):
    """
    Chuyển đổi danh sách các đa giác (polygons) thành một mặt nạ ảnh nhị phân (binary mask).
    
    Args:
        shape (tuple): Kích thước của mặt nạ cần tạo ra, thường là (Chiều_cao, Chiều_rộng, ...).
        polygons (list): Danh sách các đa giác, mỗi đa giác cấu thành từ một chuỗi các tọa độ (x, y).
        
    Returns:
        np.ndarray: Mặt nạ ảnh 1 kênh (1-channel grayscale) với giá trị 255 ở bên trong khối đa giác.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def cut_patch(image, mask, margin=15):
    """
    Hàm cắt mảng con (patch) dựa trên mặt nạ (mask) đã chọn.
    Tối ưu hóa: Sử dụng cv2.boundingRect (chạy bằng C++) thay cho np.where để tìm khung bao siêu nhanh.
    
    Args:
        image (np.ndarray): Ảnh nguồn ban đầu.
        mask (np.ndarray): Mặt nạ nhị phân của vùng lỗi cần lấy.
        margin (int, optional): Phần viền mở rộng (padding/margin) lấy thêm xung quanh để tạo dải 
                                chuyển màu, hữu ích cho các thuật toán hòa trộn (blend). Mặc định là 15.
                                
    Returns:
        tuple[np.ndarray, np.ndarray]: Trả về 1 mảng patch đã cắt và 1 mảng mask tương ứng đã cắt.
    """
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return image.copy(), mask.copy()

    H, W = image.shape[:2]
    # Lấy thêm margin để tạo thông tin nền (context) giúp chuyển viền (feather/seamless blend) tự nhiên hơn
    x1 = max(0, x - margin)
    x2 = min(W, x + w + margin)
    y1 = max(0, y - margin)
    y2 = min(H, y + h + margin)

    patch = image[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]
    return patch, patch_mask