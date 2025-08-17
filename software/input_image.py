import cv2
import re
import numpy as np

class InputImage:
	movement = 0
	offset_x = 0

	def __init__(self, buffer):
		self.source = cv2.cvtColor(buffer, cv2.COLOR_BGR2BGRA)

	# valid flashes are in a range of brigthness values
	# if the flash fired twice, this value will exceed the limit
	# if the flash did not fire, it will be below
	def valid_flash_brightness(self, offset, field, min, max):
		h, w, _ = self.source.shape

		grayscale = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
		brightness = self.brightness(grayscale, offset, offset, field)

		if brightness > min and brightness < max:
			return True

		# the pantographs sometimes touch the top edge
		# test the left pixel too, if it is in range, the image is valid
		brightness = self.brightness(grayscale, offset, h - offset, field)

		if brightness > min and brightness < max:
			return True

		return False

	def brightness(self, grayscale, x, y, field):
		field = grayscale[y-field:y+field, x-field:x+field]

		return np.mean(field)

	def rotate(self, angle, cutoff):
		(h, w) = self.source.shape[:2]
		center = (w // 2, h // 2)

		# compute the rotation matrix
		M = cv2.getRotationMatrix2D(center, angle, 1.0)

		# compute the new bounding dimensions
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])
		new_w = int((h * sin) + (w * cos))
		new_h = int((h * cos) + (w * sin))

		# adjust rotation matrix to account for translation
		M[0, 2] += (new_w / 2) - center[0]
		M[1, 2] += (new_h / 2) - center[1]

		# perform rotation
		rotated = cv2.warpAffine(self.source, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

		# crop the image to cut off black bars
		cropped = rotated[cutoff:new_h-cutoff, cutoff:new_w-cutoff]

		self.rotated = cropped

	def create_edge_mask(self, coarse_window):
		kernel_size = 3
		gray = cv2.cvtColor(self.rotated, cv2.COLOR_BGR2GRAY)

		# Get current dimensions
		height, width = gray.shape
		small_gray = cv2.resize(gray, (width // coarse_window, height // coarse_window), interpolation=cv2.INTER_AREA)

		self.edge_mask = cv2.convertScaleAbs(
			cv2.magnitude(
				cv2.Sobel(small_gray, cv2.CV_64F, 1, 0, ksize=kernel_size),
				cv2.Sobel(small_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
			)
		)

	def create_coarse_edge_mask(self, scale):
		self.coarse_edge_mask = cv2.resize(self.edge_mask, (0, 0), fx=1 / scale, fy=1 / scale)

	def create_contrast_map(self):
		# Split the image into color channels (BGRA assumed)
		b, g, r, _ = cv2.split(self.rotated)

		# Compute Laplacian for each channel
		lap_b = cv2.convertScaleAbs(cv2.Laplacian(b, cv2.CV_64F))
		lap_g = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F))
		lap_r = cv2.convertScaleAbs(cv2.Laplacian(r, cv2.CV_64F))

  		# Sum of all channels
		self.full_contrast_map = lap_b + lap_g + lap_r

		# blur a bit to break up single pixel highlights
		# boxblur by 1px (3 field)
		self.contrast_map = cv2.GaussianBlur(self.full_contrast_map, (3, 3), 0)

	def width(self):
		(h, w) = self.source.shape[:2]

		return w
