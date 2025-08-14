import numpy as np

def merge_images(images):
	frame_width = images[0].rotated.shape[1]
	frame_height = images[0].rotated.shape[0]

	output_width = frame_width + images[-1].offset_x
	output_height = frame_height

	canvas = np.zeros((output_height, output_width, 4), dtype=np.uint8)

	for x in range(0, output_width):
		overlaps = [image for image in images if x >= image.offset_x and x < image.offset_x + frame_width]

		if x % 100 == 0:
			print(x, len(overlaps))

		contrast_stack = []
		pixel_stack = []

		for layer in overlaps:
			col_index = x - layer.offset_x
			contrast_column = layer.contrast_map[:, col_index]  # (H,)
			pixel_column = layer.rotated[:, col_index, :]       # (H, 4)

			contrast_stack.append(contrast_column)
			pixel_stack.append(pixel_column)

		contrast_stack = np.stack(contrast_stack, axis=0).astype(np.float32)  # (L, H)
		pixel_stack = np.stack(pixel_stack, axis=0).astype(np.float32)        # (L, H, 4)

		# For each row, pick the top N contrasts
		top_n = min(5, len(overlaps))

		H = contrast_stack.shape[1]
		top_indices = np.argsort(-contrast_stack, axis=0)[:top_n, np.arange(H)]  # (top_n, H)

		# Gather top N pixels and contrasts
		top_pixels = np.zeros((top_n, H, 4), dtype=np.float32)
		top_contrasts = np.zeros((top_n, H), dtype=np.float32)
		for i in range(H):
			top_pixels[:, i, :] = pixel_stack[top_indices[:, i], i, :]
			top_contrasts[:, i] = contrast_stack[top_indices[:, i], i]

		# Normalize weights and blend
		weights = top_contrasts + 1e-6
		weights_sum = np.sum(weights, axis=0, keepdims=True)
		normalized_weights = weights / weights_sum
		weighted_pixels = top_pixels * normalized_weights[:, :, np.newaxis]
		blended_column = np.sum(weighted_pixels, axis=0)

		canvas[:, x] = np.clip(blended_column, 0, 255).astype(np.uint8)

	return canvas
