from time import time
import cv2
import os
import numpy as np
from uuid import uuid4

from input_image import InputImage
from movement import calculate_movement
from merger import merge_images
from slice import SliceInput

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

	images = []
	total_movement = 0

	last_capture = time()

	def __init__(self):
		self.focus_map = cv2.imread('focus-map.png', cv2.IMREAD_UNCHANGED)[:, :, 3]

	def add(self, image: InputImage):
		print('CAPTURE', time() - self.last_capture)
		self.last_capture = time()

		image.rotate(self.rotation, self.cutoff)
		image.create_edge_mask(self.coarse_window)

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

		images = [image for image in self.images]
		pool = []
		self.images = []

		for image in images:
			pool.append(SliceInput(image))
			image.offset_x -= self.slice

			if image.offset_x >= image.width():
				self.images.append(image)

		self.total_movement -= self.slice
		print('MERGE', self.total_movement, len(images), len(self.images))

		merged = merge_images(pool, self.focus_map)

		print('WRITE')
		cv2.imwrite('stitched-' + self.session + '-' + str(self.slice_index) + '.png', merged)
