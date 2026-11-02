import cv2
import numpy as np

# the flatfield is estimated on images downscaled by this factor
flatfield_scale = 8

# the flash does not light the frame evenly
# every point of the train passes through the whole frame, so it is seen at different brightnesses
# blending those frames creates smudges and bands on uniform surfaces
#
# observed = illumination(sensor position) * reflectance(train position)
# both are solved alternately in log space from all frames
def estimate_flatfield(frames, size, iterations = 8):
	height, width = size
	h, w = frames[0]['small'].shape

	min_y = min(frame['y'] for frame in frames)
	positions = [(int(round(frame['x'] / flatfield_scale)), int(round((frame['y'] - min_y) / flatfield_scale))) for frame in frames]

	scene_w = max(x for x, y in positions) + w
	scene_h = max(y for x, y in positions) + h

	logs = []
	masks = []

	for frame in frames:
		log = (np.log(frame['small'] + 1) - np.log(frame['level'])).astype(np.float32)

		# clipped, black and detailed areas do not tell anything about the illumination
		# (the position is rounded to the small scale, so edges do not line up exactly)
		gradient = cv2.magnitude(cv2.Sobel(log, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(log, cv2.CV_32F, 0, 1, ksize=3))
		mask = (frame['small_max'] < 240) & (frame['small'] > 12) & (gradient < 0.4)

		logs.append(log)
		masks.append(mask.astype(np.float32))

	illumination = np.zeros((h, w), dtype=np.float32)
	weights = masks

	for iteration in range(iterations):
		scene_sum = np.zeros((scene_h, scene_w), dtype=np.float32)
		scene_count = np.zeros((scene_h, scene_w), dtype=np.float32)

		for log, weight, (x, y) in zip(logs, weights, positions):
			scene_sum[y:y + h, x:x + w] += (log - illumination) * weight
			scene_count[y:y + h, x:x + w] += weight

		scene = scene_sum / np.maximum(scene_count, 1e-6)

		illumination_sum = np.zeros((h, w), dtype=np.float32)
		illumination_count = np.zeros((h, w), dtype=np.float32)
		residuals = []

		for log, weight, (x, y) in zip(logs, weights, positions):
			residual = log - scene[y:y + h, x:x + w]

			illumination_sum += residual * weight
			illumination_count += weight
			residuals.append(residual)

		illumination = smooth(illumination_sum, illumination_count)

		# brightness of every row stays the same, weighted like the frames are blended (center counts most)
		center = np.sin(np.pi * (np.arange(w) + 0.5) / w) ** 2
		illumination -= (illumination * center).sum(axis=1, keepdims=True) / center.sum()

		# ignore what does not fit the model (ghosts, reflections, moving shadows)
		weights = [mask * (np.abs(residual - illumination) < 0.08) for mask, residual in zip(masks, residuals)]

	print('[flatfield] illumination range ' + str(round(float(np.exp(illumination.min())), 3)) + ' - ' + str(round(float(np.exp(illumination.max())), 3)))

	return cv2.resize(np.exp(illumination), (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)

# smooth sparse samples, missing values are filled in from the surroundings
def smooth(values, counts, sigma = 2):
	values = cv2.GaussianBlur(values, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
	counts = cv2.GaussianBlur(counts, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

	filled = values / np.maximum(counts, 1e-6)

	# large holes, fill with a much wider blur
	wide_values = cv2.GaussianBlur(values, (0, 0), sigma * 8, borderType=cv2.BORDER_REPLICATE)
	wide_counts = cv2.GaussianBlur(counts, (0, 0), sigma * 8, borderType=cv2.BORDER_REPLICATE)

	confidence = np.clip(counts / (counts.max() * 0.05 + 1e-6), 0, 1)

	return confidence * filled + (1 - confidence) * wide_values / np.maximum(wide_counts, 1e-6)
