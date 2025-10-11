import cv2
import numpy as np
from requests import get, post

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
	a = 0.055
	return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
	a = 0.055
	return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)

def temperature_tint(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
	blue_gain  = 1 - temperature
	red_gain   = 1 + temperature
	green_gain = 1.0

	blue_gain  *= (1 + tint)
	red_gain   *= (1 + tint)
	green_gain *= (1 - tint)

	gains = np.array([blue_gain, green_gain, red_gain], dtype=np.float32)  # B, G, R
	return np.clip(img * gains, 0, 1)

def exposure(img: np.ndarray, stops: float) -> np.ndarray:
	lin = srgb_to_linear(img)
	lin *= 2.0 ** stops
	lin = np.clip(lin, 0, 1)

	return linear_to_srgb(lin)

def contrast(img: np.ndarray, amount: float) -> np.ndarray:
	alpha = 1.0 + amount
	out = (img - 0.5) * alpha + 0.5

	return np.clip(out, 0, 1)

def texture(img: np.ndarray, amount: float, radius: float = 2.0) -> np.ndarray:
	lin = srgb_to_linear(img)
	k = int(max(3, (radius * 6) // 2 * 2 + 1))
	blur = cv2.GaussianBlur(lin, (k, k), radius, borderType=cv2.BORDER_REFLECT101)
	high = lin - blur
	enhanced = np.clip(lin + amount * high, 0, 1)

	return linear_to_srgb(enhanced)

# ---------- pipeline ----------
def apply_filter(
	bgr_u8: np.ndarray,

	temperature_val: float = -0.10,
	tint_val: float = 0.1,

	exposure_stops: float = 1,
	contrast_amt: float = -0.1,

	texture_amt: float = 0.25,
	texture_radius: float = 2.0,
) -> np.ndarray:
	img = bgr_u8.astype(np.float32) / 255.0  # BGR in [0,1]

	img = temperature_tint(img, temperature=temperature_val, tint=tint_val)
	img = exposure(img, stops=exposure_stops)
	img = contrast(img, amount=contrast_amt)
	img = texture(img, amount=texture_amt, radius=texture_radius)

	return np.clip(img * 255.0, 0, 255).astype(np.uint8)


images = [['b79634c1-0399-41ae-bc9e-e9f39fa5d252', '91', 'forward'],
['5b2bdaff-f64f-4e5e-b582-559f3d4d2761', '6z', 'forward'],
['0e440c01-29de-4a07-b322-3800ef2565cf', '3u', 'forward'],
['be620d92-07f2-457a-8e0a-385320412c40', '91', 'forward'],
['50e1ecda-6f1e-480d-90aa-352e8350d6cc', '91', 'forward'],
['7ba47bad-0427-4e29-a887-a61fc3b17881', '8o', 'forward'],
['37eda98b-0995-4270-9c85-45918dab0f42', '3l', 'forward'],
['3ae674d4-2aa0-45b6-b28b-ee6272a7dc68', '49', 'forward'],
['67a27b25-701c-4346-90ea-65477494b195', '4a', 'forward'],
['c720e429-d417-42cc-9cae-9e7d6e9769b6', '7u', 'forward'],
['44f33fb0-cfff-423b-aa35-9558348ad464', '4e', 'forward'],
['cee7b3f2-bf8f-4195-924f-8a71165c5d63', '4t', 'forward'],
['f2151966-8733-424d-ad39-4c179403490a', '8q', 'forward'],
['e2553e01-ebc5-4404-b71f-77385c544560', '7w', 'forward'],
['697ed786-573d-4774-beab-743dfbff3ff3', '4n', 'forward'],
['3b598528-b7af-440b-b54b-7fe4ef85644f', '70', 'reverse']]

for source in images:
	print(source)
	downloaded = get('https://kalkbreite.com/capture/' + source[0] + '/full')

	image = cv2.imdecode(np.frombuffer(downloaded.content, np.uint8), cv2.IMREAD_COLOR)
	image = apply_filter(image)

	ok, encoded = cv2.imencode('.png', image)
	post('https://kalkbreite.com/capture/' + source[1] + '/' + source[2], encoded.tobytes())
