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

	slice = 10000 # target width of a slice (will be a bit bigger)
	slice_index = 0 # number of current slice
	slice_keep = 0 # will be set to width of image. how much will be kept of the last slice

	images = []
	total_movement = 0

	def add(self, image: InputImage):
		self.slice_keep = image.width()

		image.rotate(self.rotation, self.cutoff)
		image.create_edge_mask(self.coarse_window)
		image.create_contrast_map()

		if len(self.images):
			movement = calculate_movement(self.images[-1], image, self.coarse_window)
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

		print('*', self.total_movement)

		if self.total_movement > self.slice:
			self.merge_slice()

	def merge_slice(self):
		self.slice_index += 1
		print('slice', self.slice_index)

		pool = [image for image in self.images]
		self.images = []

		for image in pool:
			if image.offset_x > self.slice - self.slice_keep:
				self.images.append(image)

		print('MERGE', self.total_movement, len(pool), len(self.images))

		# merged = merge_images(pool)
		# cv2.imwrite('stitched-' + self.session + '-' + str(self.slice_index) + '.png', merged)
