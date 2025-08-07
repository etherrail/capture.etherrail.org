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

		# Stack to shape (L, H) and (L, H, 4)
		contrast_stack = np.stack(contrast_stack, axis=0).astype(np.float32)  # (L, H)
		pixel_stack = np.stack(pixel_stack, axis=0).astype(np.float32)        # (L, H, 4)

		# Avoid divide-by-zero by adding epsilon
		weights = contrast_stack + 1e-6  # (L, H)
		weights_sum = np.sum(weights, axis=0, keepdims=True)  # (1, H)
		normalized_weights = weights / weights_sum  # (L, H)

		# Apply weights to pixels
		# Expand weights to (L, H, 1) to match pixel_stack
		weighted_pixels = pixel_stack * normalized_weights[:, :, np.newaxis]  # (L, H, 4)
		blended_column = np.sum(weighted_pixels, axis=0)  # (H, 4)

		# Assign to canvas
		canvas[:, x] = np.clip(blended_column, 0, 255).astype(np.uint8)

	return canvas
