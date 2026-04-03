# ─────────────────────────────────────────────
#  core/image_augmentor.py  —  Basic Image Augmentation Engine
#  Các phương pháp augmentation cơ bản (OpenCV + NumPy)
# ─────────────────────────────────────────────

import random
import numpy as np
import cv2


# ═══════════════════════════════════════════════
#  GEOMETRIC
# ═══════════════════════════════════════════════

def aug_rotate(img: np.ndarray, angle_min: float = -30, angle_max: float = 30) -> np.ndarray:
    """Xoay ảnh một góc ngẫu nhiên."""
    h, w = img.shape[:2]
    angle = random.uniform(angle_min, angle_max)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REFLECT_101)


def aug_flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Lật ảnh ngang."""
    return cv2.flip(img, 1)


def aug_flip_vertical(img: np.ndarray) -> np.ndarray:
    """Lật ảnh dọc."""
    return cv2.flip(img, 0)


def aug_scale(img: np.ndarray, scale_min: float = 0.7, scale_max: float = 1.3) -> np.ndarray:
    """Phóng to / thu nhỏ ngẫu nhiên."""
    h, w = img.shape[:2]
    factor = random.uniform(scale_min, scale_max)
    new_w = max(4, int(w * factor))
    new_h = max(4, int(h * factor))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════
#  COLOR / INTENSITY
# ═══════════════════════════════════════════════

def aug_brightness(img: np.ndarray, delta_min: float = -40, delta_max: float = 40) -> np.ndarray:
    """Điều chỉnh độ sáng ngẫu nhiên."""
    delta = random.uniform(delta_min, delta_max)
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def aug_contrast(img: np.ndarray, factor_min: float = 0.6, factor_max: float = 1.4) -> np.ndarray:
    """Điều chỉnh độ tương phản ngẫu nhiên."""
    factor = random.uniform(factor_min, factor_max)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def aug_saturation(img: np.ndarray, factor_min: float = 0.5, factor_max: float = 1.5) -> np.ndarray:
    """Điều chỉnh độ bão hòa màu."""
    if len(img.shape) < 3 or img.shape[2] == 1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(factor_min, factor_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ═══════════════════════════════════════════════
#  NOISE & BLUR
# ═══════════════════════════════════════════════

def aug_gaussian_noise(img: np.ndarray, std_min: float = 5, std_max: float = 30) -> np.ndarray:
    """Thêm nhiễu Gaussian."""
    std = random.uniform(std_min, std_max)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_gaussian_blur(img: np.ndarray, ksize_min: int = 3, ksize_max: int = 9) -> np.ndarray:
    """Làm mờ Gaussian."""
    k = random.choice(range(ksize_min, ksize_max + 1, 2))
    return cv2.GaussianBlur(img, (k, k), 0)


# ═══════════════════════════════════════════════
#  REGISTRY — Danh sách thuật toán
# ═══════════════════════════════════════════════

AUGMENTATION_REGISTRY = [
    # ── Geometric ──
    {
        "key": "rotate", "name": "Rotation", "category": "Geometric",
        "fn": aug_rotate,
        "params": [
            ("angle_min", "Min Angle", "float", -30, -180, 0, 5),
            ("angle_max", "Max Angle", "float", 30, 0, 180, 5),
        ],
    },
    {
        "key": "flip_h", "name": "Flip Horizontal", "category": "Geometric",
        "fn": aug_flip_horizontal, "params": [],
    },
    {
        "key": "flip_v", "name": "Flip Vertical", "category": "Geometric",
        "fn": aug_flip_vertical, "params": [],
    },
    {
        "key": "scale", "name": "Scale / Resize", "category": "Geometric",
        "fn": aug_scale,
        "params": [
            ("scale_min", "Min Scale", "float", 0.7, 0.1, 1.0, 0.1),
            ("scale_max", "Max Scale", "float", 1.3, 1.0, 3.0, 0.1),
        ],
    },

    # ── Color / Intensity ──
    {
        "key": "brightness", "name": "Brightness", "category": "Color",
        "fn": aug_brightness,
        "params": [
            ("delta_min", "Δ Min", "float", -40, -100, 0, 5),
            ("delta_max", "Δ Max", "float", 40, 0, 100, 5),
        ],
    },
    {
        "key": "contrast", "name": "Contrast", "category": "Color",
        "fn": aug_contrast,
        "params": [
            ("factor_min", "Min Factor", "float", 0.6, 0.1, 1.0, 0.1),
            ("factor_max", "Max Factor", "float", 1.4, 1.0, 3.0, 0.1),
        ],
    },
    {
        "key": "saturation", "name": "Saturation", "category": "Color",
        "fn": aug_saturation,
        "params": [
            ("factor_min", "Min Factor", "float", 0.5, 0.0, 1.0, 0.1),
            ("factor_max", "Max Factor", "float", 1.5, 1.0, 3.0, 0.1),
        ],
    },

    # ── Noise & Blur ──
    {
        "key": "gaussian_noise", "name": "Gaussian Noise", "category": "Noise & Blur",
        "fn": aug_gaussian_noise,
        "params": [
            ("std_min", "Std Min", "float", 5, 1, 50, 1),
            ("std_max", "Std Max", "float", 30, 5, 100, 5),
        ],
    },
    {
        "key": "gaussian_blur", "name": "Gaussian Blur", "category": "Noise & Blur",
        "fn": aug_gaussian_blur,
        "params": [
            ("ksize_min", "Kernel Min", "int", 3, 3, 15, 2),
            ("ksize_max", "Kernel Max", "int", 9, 3, 31, 2),
        ],
    },
]


def get_registry_by_category() -> dict[str, list[dict]]:
    """Trả về registry nhóm theo category."""
    cats: dict[str, list[dict]] = {}
    for entry in AUGMENTATION_REGISTRY:
        cat = entry["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(entry)
    return cats


def run_pipeline(img: np.ndarray, enabled_keys: list[str],
                 params_override: dict[str, dict] | None = None) -> np.ndarray:
    """Chạy pipeline augmentation tuần tự."""
    if params_override is None:
        params_override = {}

    lookup = {e["key"]: e for e in AUGMENTATION_REGISTRY}
    result = img.copy()

    for key in enabled_keys:
        entry = lookup.get(key)
        if entry is None:
            continue
        fn = entry["fn"]
        kwargs = {}
        for p_name, _, p_type, p_default, *_ in entry["params"]:
            val = params_override.get(key, {}).get(p_name, p_default)
            kwargs[p_name] = int(val) if p_type == "int" else float(val)
        result = fn(result, **kwargs)

    return result
