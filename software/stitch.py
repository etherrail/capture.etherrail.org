from time import time
import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from urllib.request import urlopen

from input_image import InputImage
from movement import calculate_movement
from merger import merge_images
from slice import SliceInput
from requests import post

class Stitcher:
	session = ''

	# base rotation to align to track
	# will rotate back to isometric view when complete
	rotation = 271.2
	cutoff = 55

	# size of the coarse window for sobbel checks
	# the sobbel field is half of this
	coarse_window = 10

	slice = 10000 # target width of a slice (will be a bit bigger)
	slice_index = 0 # number of current slice

	last_image = None
	total_movement = 0

	last_capture = time()

	executor = ProcessPoolExecutor()

	def __init__(self):
		with urlopen('https://kalkbreite.com/capture/session/create') as response:
			self.session = response.read().decode('utf-8')

			print('session ready', self.session)

		self.focus_map = cv2.imread('focus-map.png', cv2.IMREAD_UNCHANGED)[:, :, 3]

	def add(self, image: InputImage):
		print('CAPTURE', time() - self.last_capture)
		self.last_capture = time()

		image.rotate(self.rotation, self.cutoff)
		image.create_edge_mask(self.coarse_window)

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

		self.last_image = image
		self.executor.submit(upload, self.session, image.rotated, image.offset_x)

def upload(session, image, offset_x):
	location = 'https://kalkbreite.com/capture/session/' + session + '/' + str(offset_x) + '/0'

	print('posting to ' + location)
	success, image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

	if success:
		post(location, data=image.tobytes())
		print('posted')
	else:
		print('Failed to encode JPEG')
