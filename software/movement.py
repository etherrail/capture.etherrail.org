import cv2
import numpy as np

# images used for the movement detection are downscaled by this factor
movement_scale = 2

# peaks closer than this to zero belong to the static scene (track, backdrop)
static_radius = 3

# number of moving peaks (besides the static one) that are tested as candidates
candidate_count = 5

# prepare a frame for the movement detection
# band-pass filtered, so that lighting gradients do not influence the match
# normalized by the flash reference, so that dark frames match normal frames
def movement_image(gray, reference):
	small = cv2.resize(gray * (128 / reference), None, fx=1 / movement_scale, fy=1 / movement_scale, interpolation=cv2.INTER_AREA)

	return small - cv2.GaussianBlur(small, (0, 0), 8)

# finds the possible movements from start to end
#
# the track and the backdrop do not move, they always create a correlation peak at zero
# when only a small part of the train is visible (entering or leaving), this peak wins and the movement is lost
# repeating structures (windows, doors) create multiple peaks of similar height
# so the strongest peaks are returned as candidates, each with the error it leaves in the changed part of the image
# the stitcher picks one using the movement of the previous frames
def movement_candidates(start, end):
	# nothing changed between the frames, the train stands still
	difference = cv2.GaussianBlur(np.abs(start - end), (0, 0), 4)
	changed = difference > 4 * np.median(difference) + 1

	if changed.mean() < 0.01:
		return [{ 'x': 0, 'y': 0, 'error': 0, 'strength': 1, 'static': True, 'still': True }]

	window = cv2.createHanningWindow(start.shape[::-1], cv2.CV_32F)
	surface = correlation_surface(start, end, window)

	candidates = []

	for index, (dx, dy, strength) in enumerate(peaks(surface, candidate_count)):
		candidates.append({
			'x': dx * movement_scale,
			'y': dy * movement_scale,
			'error': alignment_error(start, end, (dx, dy), changed),
			'strength': strength,
			'static': index == 0,
			'still': False
		})

	return candidates

def correlation_surface(start, end, window):
	a = np.fft.rfft2(start * window)
	b = np.fft.rfft2(end * window)

	cross = a * np.conj(b)
	cross /= np.abs(cross) + 1e-9

	return np.fft.fftshift(np.fft.irfft2(cross, s=start.shape))

# the static peak and the strongest count local maxima away from zero, relative to the center
def peaks(surface, count):
	h, w = surface.shape
	cy, cx = h // 2, w // 2

	static = surface[cy - static_radius:cy + static_radius + 1, cx - static_radius:cx + static_radius + 1]
	y, x = np.unravel_index(np.argmax(static), static.shape)
	result = [refine(surface, y + cy - static_radius, x + cx - static_radius)]

	maxima = (surface == cv2.dilate(surface, np.ones((5, 5), np.uint8))) & np.isfinite(surface)
	maxima[cy - static_radius:cy + static_radius + 1, cx - static_radius:cx + static_radius + 1] = False

	ys, xs = np.nonzero(maxima)
	order = np.argsort(surface[ys, xs])[::-1][:count]

	for index in order:
		result.append(refine(surface, ys[index], xs[index]))

	return result

# weighted centroid around a peak, for subpixel accuracy
def refine(surface, y, x):
	h, w = surface.shape
	cy, cx = h // 2, w // 2

	r = 2
	field = surface[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
	field = np.where(np.isfinite(field), field, 0).clip(0)
	yy, xx = np.mgrid[max(y - r, 0) - y:field.shape[0] + max(y - r, 0) - y, max(x - r, 0) - x:field.shape[1] + max(x - r, 0) - x]
	total = field.sum() + 1e-12

	return x - cx + (xx * field).sum() / total, y - cy + (yy * field).sum() / total, float(surface[y, x])

# mean difference in the changed area when end is moved back by the movement
def alignment_error(start, end, movement, mask):
	dx, dy = movement
	h, w = start.shape

	shifted = cv2.warpAffine(start, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), borderValue=np.nan)
	error = np.abs(shifted - end)
	valid = mask & np.isfinite(error)

	if valid.sum() < 100:
		return np.inf

	return float(error[valid].mean())
