import cv2
import re
import numpy as np
from time import time

class InputImage:
	movement = 0
	offset_x = 0

	def __init__(self, buffer):
		self.source = cv2.cvtColor(buffer, cv2.COLOR_BGR2BGRA)

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

	def width(self):
		(h, w) = self.rotated.shape[:2]

		return w
