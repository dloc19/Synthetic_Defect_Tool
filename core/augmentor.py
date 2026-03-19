
#  core/augmentor.py  —  Patch Augmentation──

import random
import numpy as np
import cv2

def augment_patch(patch: np.ndarray, mask: np.ndarray,
                  rotate: bool = True,
                  rotate_range: tuple[float, float] = (-30.0, 30.0),
                  flip_h: bool = False,
                  flip_v: bool = False,
                  scale: bool = False,
                  scale_range: tuple[float, float] = (0.8, 1.2),
                  brightness: bool = False,
                  brightness_range: tuple[float, float] = (-30.0, 30.0),
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Tăng cường dữ liệu (Augmentation) cho lỗi (patch) và mặt nạ (mask) đi kèm.
    Áp dụng các phép biến đổi ngẫu nhiên như lật, xoay, phóng to/thu nhỏ, thay đổi độ sáng
    nhằm tăng tính đa dạng cho dữ liệu huấn luyện.

    Trả về (augmented_patch, augmented_mask) — Cả hai sẽ có cùng kích thước đã được điều chỉnh.
    """
    h, w = patch.shape[:2]

    # 1. Lật ảnh (Flip)
    if flip_h:
        patch = cv2.flip(patch, 1) # Lật ngang
        mask  = cv2.flip(mask,  1)
    if flip_v:
        patch = cv2.flip(patch, 0) # Lật dọc
        mask  = cv2.flip(mask,  0)

    # 2. Xoay ảnh (Rotation)
    if rotate:
        angle = random.uniform(*rotate_range)
        cx, cy = w / 2.0, h / 2.0
        # Tính toán ma trận xoay bằng hàm OpenCV (Affline Transform)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        mask  = cv2.warpAffine(mask,  M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. Phóng to/Thu nhỏ (Scale)
    if scale:
        factor = random.uniform(*scale_range)
        new_w = max(4, int(w * factor))
        new_h = max(4, int(h * factor))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 4. Điều chỉnh độ sáng (Brightness)
    if brightness:
        offset = random.uniform(*brightness_range)
        # Ép kiểu sang int/float để tránh tràn số học (overflow) trước khi clip về 0-255
        patch = np.clip(patch.astype(np.float32) + offset, 0, 255).astype(np.uint8)

    return patch, mask


def random_augment(patch: np.ndarray, mask: np.ndarray,
                   config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Áp dụng phép tăng cường dữ liệu được chỉ định bởi một từ điển cấu hình (config).
    (ánh xạ trực tiếp từ các điều khiển trên GUI).

    Các khoá (keys) trong config (tất cả đều tuỳ chọn, mặc định không chạy nếu trống):
        rotate        bool: Bật/tắt xoay
        rotate_min    float (độ): Góc xoay tối thiểu
        rotate_max    float: Góc xoay tối đa
        flip_h        bool: Lật ngang
        flip_v        bool: Lật dọc
        scale         bool: Bật/tắt scale
        scale_min     float (hệ số, ví dụ: 0.8)
        scale_max     float
        brightness    bool: Bật/tắt độ sáng
        brightness_min float (giá trị thay đổi pixel, âm là tối)
        brightness_max float
    """
    return augment_patch(
        patch, mask,
        rotate=config.get("rotate", False),
        rotate_range=(config.get("rotate_min", -30.0),
                      config.get("rotate_max",  30.0)),
        flip_h=config.get("flip_h", False),
        flip_v=config.get("flip_v", False),
        scale=config.get("scale", False),
        scale_range=(config.get("scale_min", 0.8),
                     config.get("scale_max", 1.2)),
        brightness=config.get("brightness", False),
        brightness_range=(config.get("brightness_min", -30.0),
                          config.get("brightness_max",  30.0)),
    )
