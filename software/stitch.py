import cv2
import os
import numpy as np

from input_image import InputImage
from movement import calculate_movement
from merge import merge_images

def stitch(filenames):
	images = []

	# base rotation to align to track
	# will rotate back to isometric view when complete
	rotation = 271.2
	cutoff = 55

	# size of the coarse window for sobbel checks
	# the sobbel field is half of this
	coarse_movement_window = 5

	print('loading images')

	for file in filenames:
		image = InputImage(file)

		if image.valid_flash_brightness(75, 5, 80, 255):
			image.rotate(rotation, cutoff)
			image.create_edge_mask()
			image.create_contrast_map()

			images.append(image)

	images = sorted(images, key=lambda input: input.index)

	last = images[0]
	total_movement = 0

	images[0].movement = 0
	images[0].offset_x = 0

	moved_images = [images[0]]

	for next in images[1:]:
		print('processing ' + last.file_name)
		movement = calculate_movement(last.edge_mask, next.edge_mask, coarse_movement_window)

		if movement > 5:
			total_movement += movement

			next.movement = movement
			next.offset_x = total_movement

			moved_images.append(next)

		last = next

	merged = merge_images(moved_images)
	# cv2.imwrite("isometric.png", merged)

	return merged

"""
isometric = rotate_image(canvas, 17.7, 0)
cv2.imwrite("isometric.png", isometric)

M = np.array([
	[1, 0.22, 0],
	[0, 1, 0]
], dtype=np.float32)

sheared_image = cv2.warpAffine(canvas, M, (x, y))
cv2.imwrite("straight.png", sheared_image)
"""
