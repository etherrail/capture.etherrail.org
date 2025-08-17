import numpy as np

def merge_images(images, mask):
	frame_height, frame_width = images[0].image.rotated.shape[:2]
	output_width = frame_width + max(img.offset_x for img in images)
	output_height = frame_height

	# Canvas for RGBA in float [0,1]
	canvas = np.zeros((output_height, output_width, 4), dtype=np.float32)

	# Precompute mask once in float
	mask = mask.astype(np.float32)

	for img in images:
		x_offset = img.offset_x
		h, w = img.image.rotated.shape[:2]

		# Slice the canvas once
		canvas_slice = canvas[0:h, x_offset:x_offset+w]

		# Image alpha normalized
		img_rgb = img.image.rotated[:, :, :3].astype(np.float32) / 255.0
		img_alpha = img.image.rotated[:, :, 3].astype(np.float32) / 255.0

		# Effective alpha with mask
		effective_alpha = img_alpha * mask

		# Accumulate color and alpha
		np.add(canvas_slice[:, :, :3], img_rgb * effective_alpha[:, :, None], out=canvas_slice[:, :, :3])
		np.add(canvas_slice[:, :, 3], effective_alpha, out=canvas_slice[:, :, 3])

	# Normalize RGB by total alpha, avoiding division by zero
	total_alpha = canvas[:, :, 3:4]
	np.divide(canvas[:, :, :3], total_alpha, out=canvas[:, :, :3], where=total_alpha != 0)

	# Convert back to uint8
	return np.clip(canvas * 255, 0, 255).astype(np.uint8)
