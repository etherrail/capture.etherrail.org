import cv2
import numpy as np
from uuid import uuid4

from input_image import InputImage
from movement import movement_image, calculate_movement
from flatfield import estimate_flatfield, flatfield_scale
from filter import apply_filter

# frames are processed in two passes
# 1. analyze every frame: flash level, movement and the illumination of the flash
# 2. render: place every usable frame with subpixel accuracy, correct its brightness and blend the sharpest pixels
#
# the camera is telecentric, so the frames only differ by a translation
class Stitcher:
	session = str(uuid4())

	# base rotation to align to track
	# will rotate back to isometric view when complete
	rotation = 270 - 0.125
	cutoff = 10

	# flash reference field (see InputImage.reference_brightness)
	reference_offset = 15
	reference_field = 5

	# flash level relative to the median flash
	# above: the flash fired twice, the frame is ghosted and is skipped
	# between dark and double: normal frame
	# between unusable and dark: the flash fired partially, the frame is brightened and only used where no better frame exists
	double_level = 1.15
	dark_level = 0.9
	unusable_level = 0.2

	# frames closer than this to the last rendered frame add nothing (train stopped)
	min_render_step = 5

	max_vertical_shift = 20 # ignore vertical shift if over this threshold

	# sharpness weighting
	# higher sharpness_power picks the sharpest frame more aggressively
	# sharpness_floor keeps flat areas from picking random frames (which shows up as blotches)
	sharpness_power = 2
	sharpness_floor = 10

	def __init__(self):
		self.frames = []

		self.previous_movement_images = []

	# pass 1
	def add(self, path):
		image = InputImage(cv2.imread(path))
		image.rotate(self.rotation, self.cutoff)

		gray = cv2.cvtColor(image.rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
		reference = image.reference_brightness(self.reference_offset, self.reference_field)
		current = movement_image(gray, max(max(reference), 1))

		frame = {
			'path': path,
			'reference': reference,

			# movement from the previous and the second previous frame
			'movement': [self.measure(previous, current) for previous in self.previous_movement_images],

			# small images for the flat field estimation
			'small': cv2.resize(gray, None, fx=1 / flatfield_scale, fy=1 / flatfield_scale, interpolation=cv2.INTER_AREA),
			'small_max': cv2.resize(image.rotated.max(axis=2), None, fx=1 / flatfield_scale, fy=1 / flatfield_scale, interpolation=cv2.INTER_AREA)
		}

		self.frames.append(frame)
		self.size = image.rotated.shape[:2]

		self.previous_movement_images = [current] + self.previous_movement_images[:1]

	def measure(self, previous, current):
		movement_x, movement_y = calculate_movement(previous, current)

		if abs(movement_y) > self.max_vertical_shift:
			movement_y = 0

		return movement_x, movement_y

	def classify(self):
		references = np.array([frame['reference'] for frame in self.frames])
		nominal = np.median(references, axis=0)

		for frame in self.frames:
			# the pantograph can cover one corner and darken it, so the brighter corner is used
			frame['level'] = float(max(frame['reference'][0] / nominal[0], frame['reference'][1] / nominal[1]))

			if frame['level'] > self.double_level:
				frame['type'] = 'double'
			elif frame['level'] > self.dark_level:
				frame['type'] = 'normal'
			elif frame['level'] > self.unusable_level:
				frame['type'] = 'dark'
			else:
				frame['type'] = 'unusable'

			print('[flash] ' + frame['path'] + ' ' + frame['type'] + ', level = ' + str(round(frame['level'], 2)))

	# chain the movements of all usable frames
	# every frame is compared to its direct predecessor, so a failed match can not grow into a larger gap
	def locate(self):
		x = 0
		y = 0
		last = None

		for index, frame in enumerate(self.frames):
			frame['usable'] = frame['type'] in ('normal', 'dark')

			if not frame['usable']:
				continue

			if last is not None:
				if last == index - 1:
					movement_x, movement_y = frame['movement'][0]
				elif last == index - 2 and len(frame['movement']) > 1:
					# skip over a double exposure
					movement_x, movement_y = frame['movement'][1]
				else:
					print('[stitch] no reference frame for ' + frame['path'] + ', assuming standstill')
					movement_x, movement_y = 0, 0

				x += movement_x
				y += movement_y

				print('[movement] ' + frame['path'] + ' x = ' + str(round(movement_x, 1)) + ', y = ' + str(round(movement_y, 1)))

			frame['x'] = x
			frame['y'] = y

			last = index

		# select the frames to render
		last_x = None

		for frame in self.frames:
			frame['render'] = frame['usable'] and (last_x is None or frame['x'] - last_x >= self.min_render_step)

			if frame['render']:
				last_x = frame['x']

	# pass 2
	def render(self):
		self.classify()
		self.locate()

		frames = [frame for frame in self.frames if frame['render']]
		height, width = self.size

		print('[stitch] estimating flash illumination from ' + str(len(frames)) + ' frames')
		illumination = estimate_flatfield([frame for frame in frames if frame['type'] == 'normal'], self.size)

		min_y = min(frame['y'] for frame in frames)
		max_y = max(frame['y'] for frame in frames)
		max_x = max(frame['x'] for frame in frames)

		canvas_w = int(np.floor(max_x)) + width + 1
		canvas_h = int(np.floor(max_y - min_y)) + height + 1

		print('[stitch] render ' + str(len(frames)) + ' frames, width = ' + str(canvas_w) + ', height = ' + str(canvas_h))

		canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
		taper = self.taper(height, width)

		# blending happens in a buffer that moves along with the frames
		# columns left of the current frame can not change anymore and are written to the canvas
		buffer_w = width * 2
		buffer_start = 0
		color_sum = np.zeros((canvas_h, buffer_w, 3), dtype=np.float32)
		weight_sum = np.zeros((canvas_h, buffer_w), dtype=np.float32)

		def flush(end):
			end = min(end, canvas_w, buffer_start + buffer_w)
			columns = end - buffer_start

			weights = weight_sum[:, :columns, None]
			covered = weights > 1e-12

			color = np.divide(color_sum[:, :columns], weights, out=np.zeros_like(color_sum[:, :columns]), where=covered)

			canvas[:, buffer_start:end, :3] = np.clip(color, 0, 255).astype(np.uint8)
			canvas[:, buffer_start:end, 3] = covered[..., 0] * 255

			return columns

		for frame in frames:
			x = frame['x']
			y = frame['y'] - min_y

			left = int(np.floor(x))
			top = int(np.floor(y))

			if left + width > buffer_start + buffer_w:
				columns = min(flush(left), buffer_w)
				keep = buffer_w - columns

				color_sum[:, :keep] = color_sum[:, columns:]
				weight_sum[:, :keep] = weight_sum[:, columns:]
				color_sum[:, keep:] = 0
				weight_sum[:, keep:] = 0

				buffer_start = left

			image = InputImage(cv2.imread(frame['path']))
			image.rotate(self.rotation, self.cutoff, x - left, y - top)

			color, weight = self.prepare(image.rotated, frame, illumination, taper)

			bx = left - buffer_start
			color_sum[top:top + height, bx:bx + width] += color * weight[..., None]
			weight_sum[top:top + height, bx:bx + width] += weight

		flush(canvas_w)

		print('[stitch] applying filter')

		return self.filter(canvas)

	# correct the brightness of a frame and calculate how much each pixel should contribute
	def prepare(self, rotated, frame, illumination, taper):
		color = rotated.astype(np.float32) / (illumination[..., None] * frame['level'])

		# local sharpness, calculated on the corrected image so that brightness does not count as detail
		gray = cv2.resize(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
		sharpness = cv2.GaussianBlur(detail * detail, (0, 0), 6)
		sharpness = cv2.resize(sharpness, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)

		weight = taper * (sharpness + self.sharpness_floor) ** self.sharpness_power

		# clipped pixels can not be corrected
		weight[rotated.max(axis=2) >= 250] *= 1e-3

		# partial flashes are noisy and blurred by the ambient light
		# they only fill in where no proper frame is available
		if frame['type'] == 'dark':
			weight *= frame['level'] ** 4 * 1e-3

		return color, weight

	# frames are sharpest and evenly lit in the middle, fade them out to the edges
	def taper(self, height, width):
		horizontal = np.sin(np.pi * (np.arange(width) + 0.5) / width) ** 2

		fade = 32
		vertical = np.clip((np.minimum(np.arange(height), np.arange(height)[::-1]) + 0.5) / fade, 0, 1)

		return (vertical[:, None] * horizontal[None, :]).astype(np.float32)

	# the filter is applied in strips to limit memory usage
	def filter(self, canvas):
		strip = 4096
		margin = 32
		width = canvas.shape[1]

		output = np.empty_like(canvas)

		for start in range(0, width, strip):
			end = min(start + strip, width)

			a = max(start - margin, 0)
			b = min(end + margin, width)

			output[:, start:end] = apply_filter(canvas[:, a:b])[:, start - a:end - a]

		return output
