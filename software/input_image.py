import cv2
import re
import numpy as np

class InputImage:
	def __init__(self, file_name):
		self.file_name = file_name
		self.index = int(re.findall(r'\d+', file_name)[-1])

		self.load()

	def load(self):
		self.source = cv2.cvtColor(cv2.imread(self.file_name), cv2.COLOR_BGR2BGRA)

	# valid flashes are in a range of brigthness values
	# if the flash fired twice, this value will exceed the limit
	# if the flash did not fire, it will be below
	def valid_flash_brightness(self, offset, field, min, max):
		h, w, _ = self.source.shape
		grayscale = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)

		x = w - offset
		y = h - offset

		field = grayscale[y-field:y+field, x-field:x+field]
		brightness = np.mean(field)

		return brightness > min and brightness < max

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

	def create_edge_mask(self):
		kernel_size = 3
		gray = cv2.cvtColor(self.rotated, cv2.COLOR_BGR2GRAY)

		self.edge_mask = cv2.convertScaleAbs(
			cv2.magnitude(
				cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size),
				cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
			)
		)

	def create_coarse_edge_mask(self, scale):
		self.coarse_edge_mask = cv2.resize(self.edge_mask, (0, 0), fx=1 / scale, fy=1 / scale)

	def create_contrast_map(self):
		# Convert to grayscale
		grayscale = cv2.cvtColor(self.rotated, cv2.COLOR_BGRA2GRAY)

		# Compute Laplacian
		laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)

  		# Absolute values to get contrast
		self.full_contrast_map = np.abs(laplacian)

		# blur a bit to break up single pixel highlights
		# boxblur by 2px (5 field)
		self.contrast_map = cv2.blur(self.full_contrast_map, (5, 5))
