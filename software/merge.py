import numpy as np

def merge_images(images):
	frame_width = images[0].rotated.shape[1]
	frame_height = images[0].rotated.shape[0]

	output_width = frame_width + images[-1].offset_x
	output_height = frame_height

	canvas = np.zeros((output_height, output_width, 4), dtype=np.uint8)

	for x in range(0, output_width):
		overlaps = [image for image in images if x >= image.offset_x and x < image.offset_x + frame_width]

		print(x, len(overlaps))

		for y in range(0, output_height):
			max_contrast = -1

			for layer in overlaps:
				contrast = layer.contrast_map[y, x - layer.offset_x]

				if contrast > max_contrast:
					canvas[y, x] = layer.rotated[y, x - layer.offset_x]
					max_contrast = contrast

	return canvas
