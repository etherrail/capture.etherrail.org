import numpy as np
import cv2

def merge_images(images, feather_sigma = 1.0):
	frame_width = images[0].rotated.shape[1]
	frame_height = images[0].rotated.shape[0]
	output_width = frame_width + images[-1].offset_x
	output_height = frame_height

	canvas = np.zeros((output_height, output_width, 4), dtype=np.uint8)
	max_contrast_map = np.full((output_height, output_width), -np.inf, dtype=np.float32)

	rots = [img.rotated.astype(np.float32) for img in images]
	contrasts = [img.contrast_map.astype(np.float32) for img in images]
	offsets = [img.offset_x for img in images]

	for rot, contrast, offset in zip(rots, contrasts, offsets):
		x_start = offset
		x_end = offset + frame_width

		# Slice the current region
		contrast_slice = max_contrast_map[:, x_start:x_end]

		# Mask of pixels where this image is sharper
		mask = contrast > contrast_slice  # (H, W)

		# Update canvas using np.where to broadcast mask
		canvas[:, x_start:x_end] = np.where(
			mask[:, :, np.newaxis],
			rot,
			canvas[:, x_start:x_end]
		)

		# Update max contrast
		max_contrast_map[:, x_start:x_end][mask] = contrast[mask]

	if feather_sigma > 0:
		ksize = max(3, int(feather_sigma * 6) | 1)  # approximate rule: kernel ~ 6*sigma

		for c in range(4):
			canvas[:, :, c] = cv2.GaussianBlur(canvas[:, :, c], (ksize, ksize), feather_sigma)

	return canvas
