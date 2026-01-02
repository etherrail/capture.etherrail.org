import cv2
import re
import numpy as np

class InputImage:
	movement = 0 # advance from last image
	shift = 0 # up down
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

		# sometimes the pantograph might overlap
		grayscale = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
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

	def top_contrast_points(self, k=500, gauss_sigma=0.1, nms_radius=1):
		gray = cv2.cvtColor(self.rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)

		if gauss_sigma and gauss_sigma > 0:
			# choose kernel from sigma (approx): ksize = 0 lets OpenCV pick appropriate size
			gray = cv2.GaussianBlur(gray, (0, 0), gauss_sigma)

		# 2) Gradient magnitude (edge strength)
		gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
		gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
		mag = cv2.magnitude(gx, gy)

		# 3) Non-maximum suppression via dilation
		# Keep only pixels that are the local maximum within the neighborhood.
		r = max(1, int(nms_radius))
		kernel = np.ones((2*r+1, 2*r+1), np.uint8)
		mag_dilated = cv2.dilate(mag, kernel)
		nms_mask = (mag == mag_dilated)  # local peaks
		# Optional small threshold to ignore near-zero noise peaks:
		thr = 1e-6
		nms_mask &= (mag > thr)

		# 4) Gather candidate coordinates and their scores
		ys, xs = np.where(nms_mask)
		if len(xs) == 0:
			return np.empty((0, 2), dtype=int)

		scores = mag[ys, xs]

		# 5) Take top-k by score
		idx = np.argsort(scores)[::-1]
		idx = idx[:k]
		xs_top = xs[idx]
		ys_top = ys[idx]

		# Return (x, y) coordinates as ints
		pts = np.stack([xs_top, ys_top], axis=1).astype(int)
		return pts

	def create_focus_map(self, window):
		# Connect every point with every other point
		points = self.top_contrast_points()

		height, width = self.rotated.shape[:2]
		canvas = np.zeros((height, width), dtype=np.uint8)
		canvas[:] = 1

		for y in range(window, height - window):
			selection = [point[0] for point in points if point[1] > y - window and point[1] < y + window]

			if len(selection):
				size = (max(selection) - min(selection)) / 4
				middle = np.average(selection)

				cv2.circle(canvas, (int(middle), y), int(size), 255, -1)

		# add border to fade out edges
		inset = int(window * 2)

		canvas[:, :inset] = 0
		canvas[:, -inset:] = 0

		canvas = cv2.GaussianBlur(canvas, (0, 0), window)

		self.focus_map = canvas

	def create_edge_mask(self, coarse_window):
		kernel_size = 5
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
