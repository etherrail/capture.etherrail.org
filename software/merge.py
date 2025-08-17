import numpy as np

def merge_images(images, top_n=5):
	frame_width = images[0].rotated.shape[1]
	frame_height = images[0].rotated.shape[0]
	output_width = frame_width + images[-1].offset_x
	output_height = frame_height

	print('create canvas')

	canvas = np.zeros((output_height, output_width, 4), dtype=np.uint8)

	print('cast')

	rots = [img.rotated.astype(np.float32) for img in images]
	contrasts = [img.contrast_map.astype(np.float32) for img in images]
	offsets = [img.offset_x for img in images]

	print('loop')

	for x in range(output_width):
		# Determine overlapping images for this column
		overlaps_idx = [i for i, off in enumerate(offsets) if off <= x < off + frame_width]
		if not overlaps_idx:
			continue

		L = len(overlaps_idx)
		col_height = frame_height

		if x % 100 == 0:
			print(x, L)

		# Stack contrast and pixel data
		contrast_stack = np.zeros((L, col_height), dtype=np.float32)
		pixel_stack = np.zeros((L, col_height, 4), dtype=np.float32)

		for idx, i in enumerate(overlaps_idx):
			col_idx = x - offsets[i]
			contrast_stack[idx] = contrasts[i][:, col_idx]
			pixel_stack[idx] = rots[i][:, col_idx, :]

		top_k = min(top_n, L)

		# Vectorized top-N selection
		top_indices = np.argpartition(-contrast_stack, top_k-1, axis=0)[:top_k, :]
		row_indices = np.arange(col_height)
		top_pixels = pixel_stack[top_indices, row_indices[np.newaxis, :], :]
		top_contrasts = contrast_stack[top_indices, row_indices[np.newaxis, :]]

		# Blend using normalized weights
		weights = top_contrasts + 1e-6
		weights_sum = np.sum(weights, axis=0, keepdims=True)
		normalized_weights = weights / weights_sum
		blended_column = np.sum(top_pixels * normalized_weights[:, :, np.newaxis], axis=0)

		canvas[:, x] = np.clip(blended_column, 0, 255).astype(np.uint8)

	return canvas
