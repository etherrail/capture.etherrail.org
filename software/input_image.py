import cv2
import numpy as np

class InputImage:
	# flash level relative to a nominal flash (1 = normal exposure)
	# set by the stitcher once the nominal reference brightness is known
	level = 1

	def __init__(self, buffer):
		self.source = buffer

	# brightness of the static backdrop in both top corners of the sensor
	# the flash lights the backdrop, so this is a direct measure of how much flash the frame got
	# the second corner is used when the pantograph overlaps the first
	def reference_brightness(self, offset, field):
		h, w = self.source.shape[:2]
		grayscale = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)

		return (
			self.brightness(grayscale, offset, offset, field),
			self.brightness(grayscale, offset, h - offset, field)
		)

	def brightness(self, grayscale, x, y, field):
		field = grayscale[y-field:y+field, x-field:x+field]

		return np.mean(field)

	# rotate to align the track horizontally
	# the subpixel part of the placement is applied in the same warp, so the image is only interpolated once
	def rotate(self, angle, cutoff, shift_x = 0, shift_y = 0):
		(h, w) = self.source.shape[:2]
		center = (w // 2, h // 2)

		# compute the rotation matrix
		M = cv2.getRotationMatrix2D(center, angle, 1.0)

		# compute the new bounding dimensions
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])
		new_w = int(round((h * sin) + (w * cos)))
		new_h = int(round((h * cos) + (w * sin)))

		# adjust rotation matrix to account for translation
		M[0, 2] += (new_w / 2) - center[0] + shift_x
		M[1, 2] += (new_h / 2) - center[1] + shift_y

		# perform rotation
		rotated = cv2.warpAffine(self.source, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		# crop the image to cut off black bars
		self.rotated = rotated[cutoff:new_h-cutoff, cutoff:new_w-cutoff]

	def width(self):
		(h, w) = self.source.shape[:2]

		return w
