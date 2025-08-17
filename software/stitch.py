from time import time
import cv2
import os
import numpy as np
from uuid import uuid4
from requests import post

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
	coarse_window = 4

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

	def render(self, session):
		print('RENDER', len(self.images))
		self.slice_index += 1

		# merge all images
		merged = merge_images(self.images)
		cv2.imwrite('stitched-' + self.session + '-' + str(self.slice_index) + '.bmp', merged)

		shift = self.images[-1].offset_x

		for image in self.images:
			image.offset_x -= shift

		self.images = [image for image in self.images if image.offset_x >= 0]
		self.total_movement -= shift

		success, image = cv2.imencode('.png', merged)
		post('https://kalkbreite.com/capture/session/' + session + '/0/0', data=image.tobytes())
