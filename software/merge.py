import numpy as np

def merge_images(images, mask):
	frame_height, frame_width = images[0].rotated.shape[:2]
	output_width = frame_width + max(img.offset_x for img in images)
	output_height = frame_height

	# Canvas for RGBA in float [0,1]
	canvas = np.zeros((output_height, output_width, 4), dtype=np.float32)

	for img in images:
		x_offset = img.offset_x
		y_offset = 0  # no vertical offset
		h, w = img.rotated.shape[:2]

		# Image alpha normalized
		img_alpha = img.rotated[:, :, 3] / 255.0

		# Apply mask
		effective_alpha = img_alpha * mask  # shape (h, w)

		# Expand alpha for RGB multiplication
		effective_alpha_rgb = effective_alpha[:, :, np.newaxis]

		# Add weighted color
		canvas[y_offset:y_offset+h, x_offset:x_offset+w, :3] += (
			img.rotated[:, :, :3] / 255.0 * effective_alpha_rgb
		)

		# Sum alpha
		canvas[y_offset:y_offset+h, x_offset:x_offset+w, 3] += effective_alpha

	# Avoid division by zero
	total_alpha = canvas[:, :, 3:4]  # shape (H, W, 1)
	nonzero_mask = total_alpha[:, :, 0] > 0  # shape (H, W)

	# Normalize RGB by total alpha
	canvas[:, :, :3] = np.divide(
		canvas[:, :, :3], total_alpha,
		out=np.zeros_like(canvas[:, :, :3]),  # avoid NaNs
		where=total_alpha != 0
	)

	# Convert back to uint8
	canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)

	return canvas
