from os import listdir
from input_image import InputImage
from stitch import Stitcher
from requests import post
import cv2
import re
import numpy as np
from subprocess import check_output

stitcher = Stitcher()

frames = [int(re.search(r"\d+", file).group()) for file in listdir('input') if file.endswith('.bmp')]
frames.sort()

print('▄▖▗ ▌     ▄▖  ▘▜   ▄▖    ▗       ')
print('▙▖▜▘▛▌█▌▛▘▙▘▀▌▌▐   ▌ ▀▌▛▌▜▘▌▌▛▘█▌')
print('▙▖▐▖▌▌▙▖▌ ▌▌█▌▌▐▖  ▙▖█▌▙▌▐▖▙▌▌ ▙▖')
print('#' + check_output(['git','rev-parse','HEAD']).decode().strip()[0:7] + '               ▌         ')
print('Capture Stitcher')
print('')
print('input images: ' + ' '.join(str(frame) for frame in frames))
print('')

parts = []
base_offset = 0
offset_x = 0
offset_y = 0

for frame in frames:
	path = 'input/frame-' + str(frame) + '.bmp'
	image = InputImage(cv2.imread(path))

	if image.valid_flash_brightness(25, 5, 200, 250):
		stitcher.add(image)

		if stitcher.total_movement_x > 10000:
			base_offset += stitcher.images[-1].offset_x
			movement_x = base_offset
			movement_y = stitcher.total_movement_y

			print('movement_x = ' + str(movement_x) + ' movement_y = ' + str(movement_y), ' offset_x = ' + str(stitcher.images[-1].offset_x) + ' base_offset = ' + str(base_offset))

			parts.append([offset_x, offset_y, stitcher.render('')])

			offset_x = movement_x
			offset_y = movement_y

# render last image
parts.append([offset_x, offset_y, stitcher.render('')])

norm = []
min_x = 0
min_y = 0
max_x = 0
max_y = 0

for x, y, image in parts:
	x = int(x)
	y = int(y)
	h, w = image.shape[:2]

	min_x = min(min_x, x)
	min_y = min(min_y, y)
	max_x = max(max_x, x + w)
	max_y = max(max_y, y + h)

	norm.append((x, y, image))

canvas_w = max_x - min_x
canvas_h = max_y - min_y

canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
canvas[:] = np.array((0, 0, 0, 0), dtype=np.uint8)

for x, y, src in norm:
	h, w = src.shape[:2]
	cx = x - min_x
	cy = y - min_y

	dst_roi = canvas[cy:cy + h, cx:cx + w]

	alpha = src[..., 3:4].astype(np.float32) / 255.0
	inv = 1.0 - alpha

	# Blend RGB, and compute output alpha (simple "over" on alpha channel)
	dst_roi[..., 0:3] = (alpha * src[..., 0:3] + inv * dst_roi[..., 0:3]).astype(np.uint8)
	dst_roi[..., 3:4] = (np.clip(src[..., 3:4].astype(np.float32) + inv * dst_roi[..., 3:4].astype(np.float32), 0, 255)).astype(np.uint8)

location = 'https://kalkbreite.com/capture'
print('posting to ' + location)

success, image = cv2.imencode('.png', canvas)
post(location, data=image.tobytes())
print('posted')
