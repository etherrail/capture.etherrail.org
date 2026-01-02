from time import time
import cv2
import os
import numpy as np
from uuid import uuid4

from input_image import InputImage
from movement import calculate_movement
from merge import merge_images
from itertools import combinations
from filter import apply_filter

class Stitcher:
	session = str(uuid4())

	# base rotation to align to track
	# will rotate back to isometric view when complete
	rotation = 270 - 0.25
	cutoff = 10

	# size of the coarse window for sobbel checks
	# the sobbel field is half of this
	coarse_window = 6

	slice = 100000 # target width of a slice (will be a bit bigger)
	slice_index = 0 # number of current slice
	slice_keep = 0 # will be set to width of image. how much will be kept of the last slice

	images = []
	total_movement_x = 0
	total_movement_y = 0

	max_vertical_shift = 20 # ignore vertical shift if over this threshold

	def add(self, image: InputImage):
		self.slice_keep = image.width()

		image.rotate(self.rotation, self.cutoff)

		image.create_edge_mask(self.coarse_window)
		image.create_contrast_map()
		image.create_focus_map(25)

		self.images.append(image)

		if len(self.images):
			movement_x, movement_y = calculate_movement(self.images[-1], image, self.coarse_window)

			# ignore images with very minimal movement
			if movement_x < 5:
				return

			if abs(movement_y) > self.max_vertical_shift:
				movement_y = 0

			self.total_movement_x += movement_x
			self.total_movement_y += movement_y

			image.movement = movement_x
			image.offset_x = self.total_movement_x
			image.shift = self.total_movement_y

			return movement_x, movement_y
		else:
			image.movement = 0
			image.shift = 0
			image.offset_x = 0

			return 0, 0



	def render(self, session):
		print('RENDER', len(self.images))
		self.slice_index += 1

		# merge all images
		merged = merge_images(self.images)

		shift = self.images[-1].offset_x

		for image in self.images:
			image.offset_x -= shift

		self.images = [image for image in self.images if image.offset_x >= 0]
		self.total_movement_x -= shift

		filtered = apply_filter(merged)

		return filtered
