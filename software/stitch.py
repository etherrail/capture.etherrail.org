from time import time
import cv2
import os
import numpy as np
from uuid import uuid4

from input_image import InputImage
from movement import calculate_movement
from merge import merge_images

class Stitcher:
	session = str(uuid4())

	# base rotation to align to track
	# will rotate back to isometric view when complete
	rotation = 271.2
	cutoff = 55

	# size of the coarse window for sobbel checks
	# the sobbel field is half of this
	coarse_window = 10

	slice = 10000
	slice_index = 0

	last_image = None
	images = []
	total_movement = 0

	def add(self, image: InputImage):
		image.rotate(self.rotation, self.cutoff)
		image.create_edge_mask(self.coarse_window)
		image.create_contrast_map()

		if self.last_image:
			movement = calculate_movement(self.last_image, image, self.coarse_window)
			print(movement, self.total_movement)

			# ignore images with very minimal movement
			if movement < 5:
				return

			self.total_movement += movement

			image.movement = movement
			image.offset_x = self.total_movement
		else:
			image.movement = 0
			image.offset_x = 0

		self.images.append(image)

		if self.total_movement > self.slice:
			self.merge_slice()

	def merge_slice(self):
		self.slice_index += 1

		slice = [image for image in self.images]
		merged = merge_images(slice)

		cv2.imwrite('stitched-' + self.session + '-' + str(self.slice_index) + '.png', merged)
