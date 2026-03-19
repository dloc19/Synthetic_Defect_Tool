#  core/blender.py
import random
import numpy as np
import cv2

# -- OpenCV blend modes
NORMAL_CLONE    = cv2.NORMAL_CLONE
MIXED_CLONE     = cv2.MIXED_CLONE
MONO_CLONE      = cv2.MONOCHROME_TRANSFER

# -- Custom advanced blend modes
COLOR_MATCH_CLONE   = 10
ALPHA_BLEND_FEATHER = 11
PYRAMID_BLEND       = 12

BLEND_MODES = {
    "Poisson Clone (Standard)":    NORMAL_CLONE,
    "Poisson Clone (Mixed)":       MIXED_CLONE,
    "Poisson (Color Matched)":     COLOR_MATCH_CLONE,
    "Pyramid Seamless Blend":      PYRAMID_BLEND,
    "Alpha Blend (Soft Edge)":     ALPHA_BLEND_FEATHER,
    "Monochrome Transfer":         MONO_CLONE,
}

def match_colors(patch: np.ndarray, bg_roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Khớp tone màu (Lab color space) của patch với background context.
    Thuật toán: Chuyển đổi màu sang không gian LAB, sau đó căn chỉnh giá trị trung bình (mean)
    và độ lệch chuẩn (std) của vùng patch sao cho tiệm cận với vùng viền ngoài trên ảnh gốc.
    """
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # mean, std của patch (chỉ tính trong mask)
    p_mean, p_std = cv2.meanStdDev(patch_lab, mask=mask)
    
    # mean, std của background (vùng context xung quanh rìa lõi)
    # Dilate mask để lấy vùng lân cận của lỗ hổng trên ảnh nền làm mẫu màu
    kernel = np.ones((15, 15), np.uint8)
    bg_mask = cv2.dilate(mask, kernel)
    bg_mean, bg_std = cv2.meanStdDev(bg_lab, mask=bg_mask)

    # Tránh chia 0 nếu vùng không có độ lệch chuẩn (gradient)
    p_std[p_std == 0] = 1.0
    
    # Chuẩn hoá histogram
    res_lab = (patch_lab - p_mean.flatten()) * (bg_std.flatten() / p_std.flatten()) + bg_mean.flatten()
    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    
    # Trả về không gian BGR
    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)


def alpha_blend(bg_roi: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Làm mask mềm nhuyễn (feather) bằng GaussianBlur và trộn alpha pixel-by-pixel.
    Giúp tạo một vùng chuyển tiếp mờ (semi-transparent) dọc theo viền mặt nạ.
    """
    # Mềm viền (số càng lớn viền càng mờ, ảo hoá)
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    
    alpha = mask_blur.astype(np.float32) / 255.0
    if len(alpha.shape) == 2:
        alpha = np.expand_dims(alpha, axis=-1)
    
    blended = patch.astype(np.float32) * alpha + bg_roi.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

def pyramid_blend(bg_roi: np.ndarray, patch: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Hòa trộn theo tháp Laplacian (Laplacian Pyramid Blending).
    Chia tách ảnh thành nhiều tần số (độ chi tiết thấp đến cao) và trộn từ từ theo mask.
    Kỹ thuật này siêu hiệu quả, không bị lỗi sai màu nghiêm trọng như Poisson thông thường
    mà vẫn đảm bảo ranh giới siêu mượt.
    """
    # Convert dtype
    A = patch.astype(np.float32)
    B = bg_roi.astype(np.float32)
    M = mask.astype(np.float32) / 255.0
    if len(M.shape) == 2:
        M = np.repeat(M[:, :, np.newaxis], 3, axis=2)

    # 1. Tạo Gaussian Pyramids
    GA, GB, GM = A.copy(), B.copy(), M.copy()
    gpA, gpB, gpM = [GA], [GB], [GM]
    for i in range(levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(GA)
        gpB.append(GB)
        gpM.append(GM)

    # 2. Tạo Laplacian Pyramids cho A và B
    lpA = [gpA[-1]]
    lpB = [gpB[-1]]
    for i in range(levels, 0, -1):
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
        GEA = cv2.pyrUp(gpA[i], dstsize=size)
        GEB = cv2.pyrUp(gpB[i], dstsize=size)
        LA = cv2.subtract(gpA[i - 1], GEA)
        LB = cv2.subtract(gpB[i - 1], GEB)
        lpA.append(LA)
        lpB.append(LB)

    # 3. Kết hợp 2 nửa tháp bằng Mask
    LS = []
    for la, lb, gm in zip(lpA, lpB, reversed(gpM)):
        if la.shape[:2] != gm.shape[:2]:
            gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # 4. Tái cấu trúc ảnh từ tháp Laplacian kết hợp
    ls_ = LS[0]
    for i in range(1, levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])

    return np.clip(ls_, 0, 255).astype(np.uint8)


def blend_defect(ok_img: np.ndarray,
                 patch:  np.ndarray,
                 mask:   np.ndarray,
                 blend_mode: int = NORMAL_CLONE,
                 position: tuple[int, int] | None = None,
                 margin: int = 4,
                 ) -> np.ndarray:
    """
    Hàm lõi (core) chèn và hòa trộn patch lỗi vào ảnh OK.
    """
    # Đảm bảo mask là 1 kênh xám (grayscale)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    h, w = patch.shape[:2]
    H, W = ok_img.shape[:2]

    # Kiểm tra patch có lớn hơn ảnh gốc không (Cần thu nhỏ lại)
    if h >= H or w >= W:
        scale = min((H - 2 * margin) / h, (W - 2 * margin) / w) * 0.95
        new_w = max(4, int(w * scale))
        new_h = max(4, int(h * scale))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h, w = patch.shape[:2]

    # Đặt vị trí patch vào trong ảnh OK
    if position is None:
        max_x = max(margin, W - w - margin)
        max_y = max(margin, H - h - margin)
        x = random.randint(margin, max_x)
        y = random.randint(margin, max_y)
    else:
        x, y = position
        x = int(np.clip(x, margin, W - w - margin))
        y = int(np.clip(y, margin, H - h - margin))

    center = (x + w // 2, y + h // 2)

    # ─────────────────────────────────────────────────────────
    # Tối ưu hóa: Crop mặt nạ nhỏ nhất có thể bằng Bounding Box
    # Giảm viền trống bừa bãi xung quanh mask, triệt tiêu lỗi OpenCV.
    # ─────────────────────────────────────────────────────────
    mx, my, mw, mh = cv2.boundingRect(mask)
    if mw == 0 or mh == 0:
        return ok_img.copy()

    # Tính toán toạ độ mới cho tâm (center) của phần bị cắt trên ảnh gốc
    crop_center_x = x + mx + mw // 2
    crop_center_y = y + my + mh // 2
    actual_center = (crop_center_x, crop_center_y)

    # ─────────────────────────────────────────────────────────
    # Xử lý Blend nâng cao
    # ─────────────────────────────────────────────────────────
    if blend_mode in (COLOR_MATCH_CLONE, ALPHA_BLEND_FEATHER, PYRAMID_BLEND):
        result = ok_img.copy()
        
        # Region of Interest (ROI) trên ảnh nền, kích thước bằng đúng patch
        y1, y2 = y, y + h
        x1, x2 = x, x + w
        bg_roi = result[y1:y2, x1:x2]
        
        if blend_mode == COLOR_MATCH_CLONE:
            # Thuật toán: Khớp màu -> Seamless Clone thông thường
            matched_patch = match_colors(patch, bg_roi, mask)
            shrink_mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
            # Truyền mảng đã crop nhẹ vào solver
            try:
                result = cv2.seamlessClone(
                    matched_patch[my:my+mh, mx:mx+mw], 
                    result, 
                    shrink_mask[my:my+mh, mx:mx+mw], 
                    actual_center, NORMAL_CLONE)
            except cv2.error:
                result = cv2.seamlessClone(
                    matched_patch[my:my+mh, mx:mx+mw], 
                    result, 
                    mask[my:my+mh, mx:mx+mw], 
                    actual_center, NORMAL_CLONE)
                
        elif blend_mode == ALPHA_BLEND_FEATHER:
            # Thuật toán: Trộn alpha pixel bằng mặt nạ mờ
            blended_roi = alpha_blend(bg_roi, patch, mask)
            result[y1:y2, x1:x2] = blended_roi
            
        elif blend_mode == PYRAMID_BLEND:
            # Thuật toán: Laplacian Pyramid Seamless Blending
            blended_roi = pyramid_blend(bg_roi, patch, mask)
            result[y1:y2, x1:x2] = blended_roi
            
        return result

    # ─────────────────────────────────────────────────────────
    # Xử lý Blend OpenCV Default (Poisson Clone)
    # ─────────────────────────────────────────────────────────
    shrink_mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    try:
        # Cung cấp đúng phần chứa mask tight bounding box để Poisson Solver tính nhanh nhất
        result = cv2.seamlessClone(
            patch[my:my+mh, mx:mx+mw], 
            ok_img, 
            shrink_mask[my:my+mh, mx:mx+mw], 
            actual_center, blend_mode)
    except cv2.error:
        # Fallback an toàn nếu thuật toán seamless báo lỗi gradient
        try:
            result = cv2.seamlessClone(
                patch[my:my+mh, mx:mx+mw], 
                ok_img, 
                mask[my:my+mh, mx:mx+mw], 
                actual_center, blend_mode)
        except cv2.error:
            # Nếu tất cả đều fail (hiếm), fallback về Alpha Blend
            res_roi = alpha_blend(ok_img[y:y+h, x:x+w], patch, mask)
            result = ok_img.copy()
            result[y:y+h, x:x+w] = res_roi
        
    return result

def blend_defect_simple(ok_img: np.ndarray,
                        patch:  np.ndarray,
                        mask:   np.ndarray) -> np.ndarray:
    """Phiên bản đóng gói gọi thẳng (cho những tính năng cần sự đơn giản)"""
    return blend_defect(ok_img, patch, mask)