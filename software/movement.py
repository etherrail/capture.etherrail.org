import cv2
import numpy as np

# images used for the movement detection are downscaled by this factor
movement_scale = 2

# peaks closer than this to zero belong to the static scene (track, backdrop)
static_radius = 3

# prepare a frame for the movement detection
# band-pass filtered, so that lighting gradients do not influence the match
# normalized by the flash reference, so that dark frames match normal frames
def movement_image(gray, reference):
	small = cv2.resize(gray * (128 / reference), None, fx=1 / movement_scale, fy=1 / movement_scale, interpolation=cv2.INTER_AREA)

	return small - cv2.GaussianBlur(small, (0, 0), 8)

# measures the movement from start to end
#
# the track and the backdrop do not move, they always create a correlation peak at zero
# when only a small part of the train is visible (entering or leaving), this peak wins and the movement is lost
# so the best peak away from zero is tested too, and the one that lines up the changed part of the image is used
def calculate_movement(start, end):
	window = cv2.createHanningWindow(start.shape[::-1], cv2.CV_32F)
	surface = correlation_surface(start, end, window)

	best = peak(surface)
	moving = peak(surface, static_radius)

	# nothing changed between the frames, the train stands still
	difference = cv2.GaussianBlur(np.abs(start - end), (0, 0), 4)
	changed = difference > 4 * np.median(difference) + 1

	if changed.mean() < 0.01:
		return 0, 0

	if alignment_error(start, end, moving, changed) < 0.5 * alignment_error(start, end, best, changed):
		dx, dy = moving
	else:
		dx, dy = best

	return dx * movement_scale, dy * movement_scale

def correlation_surface(start, end, window):
	a = np.fft.rfft2(start * window)
	b = np.fft.rfft2(end * window)

	cross = a * np.conj(b)
	cross /= np.abs(cross) + 1e-9

	return np.fft.fftshift(np.fft.irfft2(cross, s=start.shape))

# location of the highest peak relative to the center, with subpixel accuracy
def peak(surface, exclude = 0):
	h, w = surface.shape
	cy, cx = h // 2, w // 2

	if exclude:
		surface = surface.copy()
		surface[cy - exclude:cy + exclude + 1, cx - exclude:cx + exclude + 1] = -np.inf

	y, x = np.unravel_index(np.argmax(surface), surface.shape)

	# weighted centroid around the peak
	r = 2
	field = surface[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
	field = np.where(np.isfinite(field), field, 0).clip(0)
	yy, xx = np.mgrid[max(y - r, 0) - y:field.shape[0] + max(y - r, 0) - y, max(x - r, 0) - x:field.shape[1] + max(x - r, 0) - x]
	total = field.sum() + 1e-12

	return x - cx + (xx * field).sum() / total, y - cy + (yy * field).sum() / total

# mean difference in the changed area when end is moved back by the movement
def alignment_error(start, end, movement, mask):
	dx, dy = movement
	h, w = start.shape

	shifted = cv2.warpAffine(start, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), borderValue=np.nan)
	error = np.abs(shifted - end)
	valid = mask & np.isfinite(error)

	if valid.sum() < 100:
		return np.inf

	return error[valid].mean()
