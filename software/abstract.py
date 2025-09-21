import cv2
import numpy as np

img_bgr = cv2.imread('stitched-0fd69afb-bf4f-4d11-a25e-f053e31c072b-1.png')

target_degrees=(33, 54)
deg_tol=2.0
canny_low=10
canny_high=150
hough_rho=1
hough_theta=np.pi/180
hough_thresh=20
min_line_len=30
max_line_gap=10
draw_thickness=2

# 1) edges
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, canny_low, canny_high, L2gradient=True)

# 2) Hough lines (probabilistic: gives endpoints)
lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_thresh,
						minLineLength=min_line_len, maxLineGap=max_line_gap)

h, w = img_bgr.shape[:2]
canvas = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

if lines is None:
	exit(0)

# Normalize target degrees into [0, 180)
targets = [((deg % 180) + 180) % 180 for deg in target_degrees]

# Helper to normalize any angle into [0, 180)
def norm180(a):
	a = a % 180.0
	return a if a >= 0 else a + 180.0

# Pick distinct colors for each target
# (B, G, R)
palette = [
	(0, 0, 255),   # red
	(0, 255, 0),   # green
	(255, 0, 0),   # blue
	(0, 255, 255), # yellow
	(255, 0, 255), # magenta
	(255, 255, 0), # cyan
]

for seg in lines:
	x1, y1, x2, y2 = seg[0]
	dx, dy = (x2 - x1), (y2 - y1)

	if dx == 0 and dy == 0:
		continue

	# Angle of the segment relative to +x axis; arctan2 returns [-180, 180]
	ang = np.degrees(np.arctan2(dy, dx))
	ang = norm180(ang)  # -> [0, 180)

	# Also consider that a line at θ is the same orientation as θ+180,
	# but after norm180 it’s already folded into [0,180).
	# Check distance to each target, also symmetric around 180.
	for idx, t in enumerate(targets):
		# minimal circular distance on the 180° circle
		# d = min(abs(ang - t), 180 - abs(ang - t))
		# if d <= deg_tol:
		color = palette[idx % len(palette)]
		cv2.line(canvas, (x1, y1), (x2, y2), color, draw_thickness, cv2.LINE_AA)
		# break  # stop after first matching target to avoid double-draw

cv2.imwrite('technical_drawing.png', canvas)
