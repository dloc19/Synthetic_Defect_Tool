import os
import uuid
from typing import Generator

import cv2
import numpy as np

from core.blender  import blend_defect, BLEND_MODES
from core.augmentor import random_augment


def generate_batch(
        ok_img:     np.ndarray,
        patch:      np.ndarray,
        mask:       np.ndarray,
        count:      int                    = 10,
        blend_mode: int | None            = None,
        aug_config: dict | None           = None,
        output_dir: str | None            = None,
        save_mask:  bool                  = False,
        prefix:     str                   = "synth",
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Tạo ra một loạt (batch) gồm `count` ảnh tổng hợp (synthetic), trả về luồng trạng thái
    của từ khoá yield (idx, result) cho phép tiến trình cập nhật từng ảnh một mà không
    làm đơ luồng hệ thống.

    Tham số
    ----------
    ok_img      Ảnh nền BGR mẫu chuẩn (mẫu tốt/OK)
    patch       Mảng ảnh lỗi (defect) hệ BGR
    mask        Mặt nạ phân vùng lỗi (chỉ có 1 kênh xám)
    count       Số lượng ảnh mong muốn sinh ra (Mặc định: 10)
    blend_mode  ID loại hiệu ứng hòa trộn sẽ dùng (Mặc định: NORMAL_CLONE)
    aug_config  Từ điển cấu hình chạy (augmentor.random_augment), bỏ qua nếu bằng None
    output_dir  Đường dẫn tải xuống, nếu có truyền vào thì sẽ tự động lưu xuống tệp dạng PNG
    save_mask   Kích hoạt cờ này để đồng thời lưu mask của phần lỗi đã sinh kèm theo
    prefix      Ký tự tiếp đầu ngữ cho tên tệp lưu (ví dụ: "synth")

    Trả về (Yields)
    ------
    (idx, result_bgr): idx là số thứ tự bắt đầu từ 1, result_bgr là mảng ma trận ảnh đầu ra
    """
    from core.blender import NORMAL_CLONE
    if blend_mode is None:
        blend_mode = NORMAL_CLONE

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i in range(1, count + 1):
        # 1. Biến đổi dữ liệu (Augment patch)
        if aug_config:
            p_aug, m_aug = random_augment(patch.copy(), mask.copy(), aug_config)
        else:
            p_aug, m_aug = patch.copy(), mask.copy()

        # 2. Hoà trộn (Blend defect)
        result = blend_defect(ok_img, p_aug, m_aug, blend_mode=blend_mode)

        # 3. Lưu trữ nếu có chỉ định thư mục (Save logic)
        if output_dir:
            # Sinh chuỗi hex tuỳ ý thông qua uuid để đảm bảo tên độc nhất
            name = f"{prefix}_{i:04d}_{uuid.uuid4().hex[:6]}"
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), result)
            # if save_mask:
            #     # Lưu mặt nạ annotator ra file PNG
            #     cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), m_aug)

        yield i, result
