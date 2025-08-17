import numpy as np

def merge_images(images, block_size):
	frame_width = images[0].rotated.shape[1]
	frame_height = images[0].rotated.shape[0]

	output_width = frame_width + images[-1].offset_x
	output_height = frame_height

	canvas = np.zeros((output_height, output_width, 4), dtype=np.uint8)

	for x in range(0, output_width, block_size):
		overlaps = [image for image in images if x < image.offset_x + frame_width and x + block_size > image.offset_x]

		if x % 100 == 0:
			print(x, len(overlaps))

		for y in range(0, output_height, block_size):
			# Canvas block coordinates
			x0, x1 = x, min(x + block_size, output_width)
			y0, y1 = y, min(y + block_size, output_height)

			if not overlaps:
				continue

			best_avg_contrast = -np.inf
			best_layer = None
			best_lx0, best_lx1 = None, None

			# Pick the layer with the highest average contrast over the block
			for layer in overlaps:
				# Layer coordinates overlapping the canvas block
				lx0 = max(0, x0 - layer.offset_x)
				lx1 = min(frame_width, x1 - layer.offset_x)
				ly0 = y0
				ly1 = y1

				if lx0 >= lx1 or ly0 >= ly1:
					continue

				contrast_block = layer.contrast_map[ly0:ly1, lx0:lx1]
				avg_contrast = np.mean(contrast_block)

				if avg_contrast > best_avg_contrast:
					best_avg_contrast = avg_contrast
					best_layer = layer
					best_lx0, best_lx1 = lx0, lx1

			if best_layer is None:
				continue

			# Compute the exact size of the block to copy
			canvas_x0 = max(x0, best_layer.offset_x)
			canvas_x1 = min(x1, best_layer.offset_x + frame_width)
			canvas_y0 = y0
			canvas_y1 = y1

			layer_x0 = canvas_x0 - best_layer.offset_x
			layer_x1 = canvas_x1 - best_layer.offset_x
			layer_y0 = canvas_y0
			layer_y1 = canvas_y1

			canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1, :] = best_layer.rotated[layer_y0:layer_y1, layer_x0:layer_x1, :]

	return canvas
