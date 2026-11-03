import cv2
import numpy as np

from input_image import InputImage

# every session starts or ends with frames of nothing but the backdrop
# they are used to recognize the backdrop behind the train and replace it with a uniform color
class Background:
	# frames that differ less than this (mean of the small grayscale) from their predecessor are identical
	still_difference = 2.5

	# empty frames are still frames within this fraction of the brightest frame of the session
	empty_brightness = 0.97

	# a pixel matches the backdrop if all channels are within match_tolerance, fades out until mismatch_tolerance
	# (light reflected from the train tints the backdrop a bit, so this can not be too tight)
	match_tolerance = 0.06
	mismatch_tolerance = 0.15

	# a canvas pixel is backdrop if this fraction of the frames covering it saw the backdrop there
	coverage_low = 0.8
	coverage_high = 0.95

	# backdrop areas smaller than this (pixels at a quarter scale) are kept, they are most likely part of the train
	min_area = 2000

	# holes in the backdrop smaller than this (pixels at a quarter scale) are filled
	max_hole = 200

	# None = median color of the backdrop
	color = None

	max_frames = 16

	def __init__(self, frames, rotation, cutoff):
		self.frames = self.find_empty_frames(frames)

		if not len(self.frames):
			print('[background] no empty frames found, backdrop is kept as is')
			self.image = None

			return

		print('[background] using ' + str(len(self.frames)) + ' empty frames: ' + ' '.join(frame['path'] for frame in self.frames))

		total = None

		for frame in self.frames:
			image = InputImage(cv2.imread(frame['path']))
			image.rotate(rotation, cutoff)

			corrected = image.rotated.astype(np.float32) / frame['level']
			total = corrected if total is None else total + corrected

		self.image = total / len(self.frames)

		# only the bright and smooth backdrop is replaced, never the track or anything else that is static
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		gradient = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), (0, 0), 4)

		region = (gray > 0.6 * np.percentile(gray, 99)) & (gradient < 0.04 * gray)
		self.region = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)

	def find_empty_frames(self, frames):
		candidates = [frame for frame in frames if frame['type'] == 'normal' and frame['still']]

		if not len(candidates):
			return []

		# compared to all frames, a slow dark wagon can also be still
		brightness = lambda frame: frame['small'].mean() / frame['level']
		brightest = max(brightness(frame) for frame in frames if frame['type'] == 'normal')

		empty = [frame for frame in candidates if brightness(frame) >= self.empty_brightness * brightest]

		# a single still frame might be a coincidence
		if len(empty) < 2:
			return []

		step = max(1, len(empty) // self.max_frames)

		return empty[::step][:self.max_frames]

	# how much each pixel of a frame looks like the backdrop, 0 - 1
	def likeness(self, rotated, level):
		if self.image is None:
			return None

		# smoothed, so that sensor noise does not count as a mismatch
		ratio = cv2.GaussianBlur(rotated.astype(np.float32) / (self.image * level + 1), (0, 0), 3)
		deviation = np.abs(ratio - 1).max(axis=2)

		likeness = np.clip((self.mismatch_tolerance - deviation) / (self.mismatch_tolerance - self.match_tolerance), 0, 1)

		return likeness * self.region

	# replace the backdrop of the finished image
	# coverage: fraction of the frames that saw the backdrop at a canvas pixel, 0 - 255
	def apply(self, canvas, coverage):
		if self.image is None:
			return canvas

		height, width = coverage.shape
		scale = 4

		small = cv2.resize(coverage, (width // scale, height // scale), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
		mask = np.clip((small - self.coverage_low) / (self.coverage_high - self.coverage_low), 0, 1)

		# drop small areas
		count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0.5).astype(np.uint8), connectivity=8)
		keep = np.zeros(count, dtype=bool)
		keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_area
		mask *= keep[labels]

		# fill small holes (dust on the sensor or the backdrop)
		count, labels, stats, _ = cv2.connectedComponentsWithStats((mask <= 0.5).astype(np.uint8), connectivity=4)
		holes = np.zeros(count, dtype=bool)
		holes[1:] = stats[1:, cv2.CC_STAT_AREA] < self.max_hole
		mask[holes[labels]] = 1

		mask = cv2.GaussianBlur(mask, (0, 0), 1)

		color = self.color

		if color is None:
			selected = cv2.resize((mask > 0.99).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
			color = np.median(canvas[..., :3][selected][::97], axis=0)

		print('[background] replacing ' + str(round(float(mask.mean()) * 100, 1)) + '% of the image with ' + str(np.round(color).astype(int).tolist()))

		mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

		# blend in strips to limit memory usage
		strip = 4096

		for start in range(0, width, strip):
			end = min(start + strip, width)

			alpha = mask[:, start:end, None]
			part = canvas[:, start:end, :3].astype(np.float32)

			canvas[:, start:end, :3] = np.clip(part * (1 - alpha) + np.array(color, dtype=np.float32) * alpha, 0, 255).astype(np.uint8)

		return canvas
