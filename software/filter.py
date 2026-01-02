import cv2
import numpy as np

def _split_color_alpha(img: np.ndarray):
    """Return (color[BGR], alpha[H,W,1] or None)."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3], img[..., 3:4]
    return img, None

def _merge_color_alpha(color: np.ndarray, alpha: np.ndarray | None):
    if alpha is None:
        return color
    return np.concatenate([color, alpha], axis=2)

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)

def temperature_tint(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    color, alpha = _split_color_alpha(img)

    blue_gain  = 1 - temperature
    red_gain   = 1 + temperature
    green_gain = 1.0

    blue_gain  *= (1 + tint)
    red_gain   *= (1 + tint)
    green_gain *= (1 - tint)

    gains = np.array([blue_gain, green_gain, red_gain], dtype=np.float32).reshape(1,1,3)  # B,G,R
    out_color = np.clip(color * gains, 0, 1)

    return _merge_color_alpha(out_color, alpha)

def exposure(img: np.ndarray, stops: float) -> np.ndarray:
    color, alpha = _split_color_alpha(img)

    lin = srgb_to_linear(color)
    lin *= 2.0 ** stops
    lin = np.clip(lin, 0, 1)
    out_color = linear_to_srgb(lin)

    return _merge_color_alpha(out_color, alpha)

def contrast(img: np.ndarray, amount: float) -> np.ndarray:
    color, alpha = _split_color_alpha(img)

    alpha_c = 1.0 + amount
    out_color = (color - 0.5) * alpha_c + 0.5
    out_color = np.clip(out_color, 0, 1)

    return _merge_color_alpha(out_color, alpha)

def texture(img: np.ndarray, amount: float, radius: float = 2.0) -> np.ndarray:
    color, alpha = _split_color_alpha(img)

    lin = srgb_to_linear(color)
    k = int(max(3, (radius * 6) // 2 * 2 + 1))
    blur = cv2.GaussianBlur(lin, (k, k), radius, borderType=cv2.BORDER_REFLECT101)
    high = lin - blur
    enhanced = np.clip(lin + amount * high, 0, 1)
    out_color = linear_to_srgb(enhanced)

    return _merge_color_alpha(out_color, alpha)

# ---------- pipeline ----------
def apply_filter(
    bgr_or_bgra_u8: np.ndarray,

    temperature_val: float = -0.10,
    tint_val: float = 0.1,

    exposure_stops: float = 1,
    contrast_amt: float = -0.1,

    texture_amt: float = 0.25,
    texture_radius: float = 2.0,
) -> np.ndarray:
    """
    Accepts uint8 BGR or BGRA. Returns the same channel count as input.
    Alpha (if present) is passed through unchanged.
    """
    # Normalize to [0,1], keep channel count
    img = bgr_or_bgra_u8.astype(np.float32) / 255.0

    img = temperature_tint(img, temperature=temperature_val, tint=tint_val)
    img = exposure(img, stops=exposure_stops)
    img = contrast(img, amount=contrast_amt)
    img = texture(img, amount=texture_amt, radius=texture_radius)

    # Clip & convert back to uint8
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return out
