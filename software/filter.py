import cv2
import numpy as np

def apply_filter(img):
	# ----- Brightness (-100%) -----
	# Scale brightness: -100% means very dark. We'll just multiply by a small factor.
	brightness_factor = 0.0  # -100% → basically black, but keeping detail we can set to 0.2
	bright_img = img * brightness_factor
	bright_img = np.clip(bright_img, 0, 1)

	# ----- Highlights (+100%) -----
	# Boost bright areas
	hls = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HLS)
	hls[:,:,1] = np.clip(hls[:,:,1] * 1.5, 0, 1)  # lightness channel
	highlight_img = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

	# ----- Shadows (+100%) -----
	# Lift darks
	gamma = 0.5  # <1 brightens shadows
	shadow_img = np.power(highlight_img, gamma)

	# ----- Texture (+50%) = Unsharp Mask for local contrast -----
	blur = cv2.GaussianBlur(shadow_img, (0,0), 3)
	texture_img = cv2.addWeighted(shadow_img, 1.5, blur, -0.5, 0)

	# ----- Saturation (+25%) -----
	hsv = cv2.cvtColor(texture_img, cv2.COLOR_BGR2HSV)
	hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.25, 0, 1)  # saturation
	sat_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# ----- Sharpen (Radius=1px, Intensity=200%) -----
	blur_sharp = cv2.GaussianBlur(sat_img, (0,0), 1)
	sharpened = cv2.addWeighted(sat_img, 2.0, blur_sharp, -1.0, 0)

	# Convert back to uint8
	output = np.clip(sharpened * 255, 0, 255).astype(np.uint8)

	return output
