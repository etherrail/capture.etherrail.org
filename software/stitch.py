import cv2
import numpy as np
from uuid import uuid4

from input_image import InputImage
from movement import movement_image, movement_candidates
from flatfield import estimate_flatfield, flatfield_scale
from filter import apply_filter
from background import Background

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

	# trains only move in one direction (top to bottom in the sensor, positive x after the rotation)
	# a stalling locomotive can rattle back a bit, anything further back is a mismatch
	direction = 1
	max_backward = 10

	# the speed is mostly constant, so candidates are weighted by how far they deviate from the expected movement
	# cost = alignment error relative to the best candidate
	#      + speed_weight * deviation / (expected speed + speed_floor)
	#      + peak_weight * ln(strongest moving peak / peak)
	# the frame interval jitters, so a much stronger peak has to be able to beat the expected speed
	# the static peak is always strong while the train is only partially visible, so it is only judged by error and speed
	# the expected movement is the median of the last speed_history movements (per frame)
	# below speed_floor the train is starting or arriving and the speed is not used
	speed_weight = 1
	speed_floor = 20
	speed_history = 3
	peak_weight = 0.5

	# sharpness weighting
	# higher sharpness_power picks the sharpest frame more aggressively
	# sharpness_floor keeps flat areas from picking random frames (which shows up as blotches)
	sharpness_power = 2
	sharpness_floor = 10

	# uncovered pixels at the top and bottom are filled in and blurred with this sigma
	# the blur fades in over extend_fade pixels from the last covered pixel
	extend_blur = 8
	extend_fade = 24

	def __init__(self):
		self.frames = []

		self.previous_movement_images = []
		self.previous_small = None

	# pass 1
	def add(self, path):
		image = InputImage(cv2.imread(path))
		image.rotate(self.rotation, self.cutoff)

		gray = cv2.cvtColor(image.rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
		reference = image.reference_brightness(self.reference_offset, self.reference_field)
		current = movement_image(gray, max(max(reference), 1))

		small = cv2.resize(gray, None, fx=1 / flatfield_scale, fy=1 / flatfield_scale, interpolation=cv2.INTER_AREA)

		frame = {
			'path': path,
			'reference': reference,

			# identical to the previous frame (used to find the empty frames)
			'still': self.previous_small is not None and np.abs(small - self.previous_small).mean() < Background.still_difference,

			# movement candidates from the previous and the second previous frame
			'movement': [self.measure(previous, current) for previous in self.previous_movement_images],

			# small images for the flat field estimation
			'small': small,
			'small_max': cv2.resize(image.rotated.max(axis=2), None, fx=1 / flatfield_scale, fy=1 / flatfield_scale, interpolation=cv2.INTER_AREA)
		}

		self.frames.append(frame)
		self.size = image.rotated.shape[:2]

		self.previous_movement_images = [current] + self.previous_movement_images[:1]
		self.previous_small = small

	def measure(self, previous, current):
		candidates = movement_candidates(previous, current)

		for candidate in candidates:
			if abs(candidate['y']) > self.max_vertical_shift:
				candidate['y'] = 0

		return candidates

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
		history = []

		for index, frame in enumerate(self.frames):
			frame['usable'] = frame['type'] in ('normal', 'dark')

			if not frame['usable']:
				continue

			if last is not None:
				# frames in between (double exposures) are skipped, the movement spans all of them
				gap = index - last
				expected = self.expected_movement(history)

				if gap <= len(frame['movement']):
					movement_x, movement_y = self.pick(frame, frame['movement'][gap - 1], None if expected is None else expected * gap)
				elif expected is not None:
					print('[stitch] no reference frame for ' + frame['path'] + ', assuming constant speed')
					movement_x, movement_y = expected * gap, 0
				else:
					print('[stitch] no reference frame for ' + frame['path'] + ', assuming standstill')
					movement_x, movement_y = 0, 0

				history.append(movement_x / gap)

				x += movement_x
				y += movement_y

			frame['x'] = x
			frame['y'] = y

			last = index

		# select the frames to render
		last_x = None

		for frame in self.frames:
			frame['render'] = frame['usable'] and (last_x is None or frame['x'] - last_x >= self.min_render_step)

			if frame['render']:
				last_x = frame['x']

	# expected movement per frame along the track, None if the train is not moving steadily
	def expected_movement(self, history):
		if not len(history):
			return None

		expected = float(np.median(history[-self.speed_history:]))

		if abs(expected) < self.speed_floor:
			return None

		return expected

	# choose the movement candidate that aligns best and is closest to the expected movement
	def pick(self, frame, candidates, expected):
		for candidate in candidates:
			candidate['allowed'] = candidate['x'] * self.direction >= -self.max_backward and np.isfinite(candidate['error'])

		allowed = [candidate for candidate in candidates if candidate['allowed']]
		best_error = max(min((candidate['error'] for candidate in allowed), default=0), 1e-6)
		best_peak = max((candidate['strength'] for candidate in allowed if not candidate['static']), default=1)

		for candidate in candidates:
			candidate['cost'] = np.inf

			if candidate['allowed']:
				candidate['cost'] = candidate['error'] / best_error

				if expected is not None:
					candidate['cost'] += self.speed_weight * abs(candidate['x'] - expected) / (abs(expected) + self.speed_floor)

				if not candidate['static']:
					candidate['cost'] += self.peak_weight * np.log(best_peak / max(candidate['strength'], 1e-6))

		picked = min(allowed, key=lambda candidate: candidate['cost'], default=None)

		print('[movement] ' + frame['path'] + ', expected x = ' + ('-' if expected is None else str(round(expected, 1))))

		for candidate in candidates:
			print('[movement]   ' + ('> ' if candidate is picked else '  ') + 'x = ' + str(round(candidate['x'], 1)).rjust(7) + ', y = ' + str(round(candidate['y'], 1)).rjust(5) + ', peak = ' + str(round(candidate['strength'], 3)).ljust(5) + ', error = ' + str(round(candidate['error'], 2)).ljust(5) + ', cost = ' + str(round(candidate['cost'], 2)) + (' (still)' if candidate['still'] else ' (static)' if candidate['static'] else '') + ('' if candidate['allowed'] else ' (backward)' if np.isfinite(candidate['error']) else ' (no overlap)'))

		if picked is None:
			fallback = 0 if expected is None else expected

			print('[movement]   no usable candidate, using x = ' + str(round(fallback, 1)))

			return fallback, 0

		print('[movement]   picked x = ' + str(round(picked['x'], 1)) + ', y = ' + str(round(picked['y'], 1)) + ', deviation = ' + ('-' if expected is None else str(round(picked['x'] - expected, 1))))

		return picked['x'], picked['y']

	# pass 2
	def render(self):
		self.classify()
		self.locate()

		frames = [frame for frame in self.frames if frame['render']]
		height, width = self.size

		print('[stitch] estimating flash illumination from ' + str(len(frames)) + ' frames')
		illumination = estimate_flatfield([frame for frame in frames if frame['type'] == 'normal'], self.size)

		background = Background(self.frames, self.rotation, self.cutoff)

		min_y = min(frame['y'] for frame in frames)
		max_y = max(frame['y'] for frame in frames)
		max_x = max(frame['x'] for frame in frames)

		canvas_w = int(np.floor(max_x)) + width + 1
		canvas_h = int(np.floor(max_y - min_y)) + height + 1

		print('[stitch] render ' + str(len(frames)) + ' frames, width = ' + str(canvas_w) + ', height = ' + str(canvas_h))

		canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
		coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
		taper = self.taper(height, width)

		# blending happens in a buffer that moves along with the frames
		# columns left of the current frame can not change anymore and are written to the canvas
		buffer_w = width * 2
		buffer_start = 0
		color_sum = np.zeros((canvas_h, buffer_w, 3), dtype=np.float32)
		weight_sum = np.zeros((canvas_h, buffer_w), dtype=np.float32)
		likeness_sum = np.zeros((canvas_h, buffer_w), dtype=np.float32)

		def flush(end):
			end = min(end, canvas_w, buffer_start + buffer_w)
			columns = end - buffer_start

			weights = weight_sum[:, :columns, None]
			covered = weights > 1e-12

			color = np.divide(color_sum[:, :columns], weights, out=np.zeros_like(color_sum[:, :columns]), where=covered)

			canvas[:, buffer_start:end, :3] = np.clip(color, 0, 255).astype(np.uint8)
			canvas[:, buffer_start:end, 3] = covered[..., 0] * 255

			likeness = np.divide(likeness_sum[:, :columns], weights[..., 0], out=np.zeros_like(likeness_sum[:, :columns]), where=covered[..., 0])
			coverage[:, buffer_start:end] = np.clip(likeness * 255, 0, 255).astype(np.uint8)

			return columns

		for frame in frames:
			x = frame['x']
			y = frame['y'] - min_y

			left = int(np.floor(x))
			top = int(np.floor(y))

			if left + width > buffer_start + buffer_w:
				columns = min(flush(left), buffer_w)
				keep = buffer_w - columns

				for buffer in (color_sum, weight_sum, likeness_sum):
					buffer[:, :keep] = buffer[:, columns:]
					buffer[:, keep:] = 0

				buffer_start = left

			image = InputImage(cv2.imread(frame['path']))
			image.rotate(self.rotation, self.cutoff, x - left, y - top)

			color, weight = self.prepare(image.rotated, frame, illumination, taper)
			likeness = background.likeness(image.rotated, frame['level'])

			bx = left - buffer_start
			color_sum[top:top + height, bx:bx + width] += color * weight[..., None]
			weight_sum[top:top + height, bx:bx + width] += weight

			if likeness is not None:
				likeness_sum[top:top + height, bx:bx + width] += likeness * weight

		flush(canvas_w)

		self.extend(canvas, coverage)

		print('[stitch] applying filter')
		canvas = self.filter(canvas)

		return background.apply(canvas, coverage)

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

	# vertical movement leaves uncovered pixels at the top and bottom
	# they are filled with the nearest covered pixel of their column, blurred more the further away they are
	# covered pixels are never changed
	def extend(self, canvas, coverage):
		strip = 4096
		margin = 3 * self.extend_blur
		height, width = coverage.shape
		rows = np.arange(height)[:, None]

		for start in range(0, width, strip):
			end = min(start + strip, width)

			a = max(start - margin, 0)
			b = min(end + margin, width)

			covered = canvas[:, a:b, 3] > 0

			if covered[:, start - a:end - a].all():
				continue

			has_pixels = covered.any(axis=0)

			if not has_pixels.any():
				continue

			# columns without any pixel (the rounded up last column) copy the nearest column with pixels
			with_pixels = np.where(has_pixels)[0]
			nearest = with_pixels[np.clip(np.searchsorted(with_pixels, np.arange(b - a)), 0, len(with_pixels) - 1)]
			before = with_pixels[np.clip(np.searchsorted(with_pixels, np.arange(b - a)) - 1, 0, len(with_pixels) - 1)]
			nearest = np.where(np.abs(before - np.arange(b - a)) < np.abs(nearest - np.arange(b - a)), before, nearest)

			first = np.argmax(covered, axis=0)[nearest]
			last = (height - 1 - np.argmax(covered[::-1], axis=0))[nearest]

			source = np.clip(rows, first[None, :], last[None, :])
			columns = nearest[None, :]

			filled = canvas[:, a:b][source, columns]
			filled_coverage = coverage[:, a:b][source, columns]

			distance = np.maximum(first[None, :] - rows, rows - last[None, :]).clip(0).astype(np.float32)

			blurred = cv2.GaussianBlur(filled[..., :3].astype(np.float32), (0, 0), self.extend_blur)
			amount = np.clip(distance / self.extend_fade, 0, 1)[..., None]

			color = filled[..., :3] * (1 - amount) + blurred * amount

			# only write the strip itself, the margin is only there for the blur
			part = slice(start - a, end - a)
			empty = ~covered[:, part]

			canvas[:, start:end, :3][empty] = np.clip(color[:, part], 0, 255).astype(np.uint8)[empty]
			canvas[:, start:end, 3][empty] = 255
			coverage[:, start:end][empty] = filled_coverage[:, part][empty]

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
