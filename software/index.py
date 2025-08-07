from os import listdir, read
from sys import stdin
from stitch import stitch
from requests import post
import cv2
import numpy as np
import time

while True:
	# tag = input('TAG: ')
	# direction = input('R/F: ')

	images = []

	for file in listdir('input'):
		if file.endswith('.bmp'):
			images.append('input/' + file)

	start = time.time()
	# stitched = stitch(images)
	#
	left = cv2.imread(images[0])
	right = cv2.imread(images[1])

	def stitch_translation(img1, img2):
		# Detect ORB features and descriptors
		orb = cv2.ORB_create(500)
		kp1, des1 = orb.detectAndCompute(img1, None)
		kp2, des2 = orb.detectAndCompute(img2, None)

		# Match features using brute-force matcher
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(des1, des2)
		matches = sorted(matches, key=lambda x: x.distance)

		# Use top matches to estimate translation
		src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

		# Estimate translation (affine with no rotation)
		shift = np.mean(dst_pts - src_pts, axis=0)
		dx, dy = shift.astype(int)

		print(dx, dy)

		# Create output canvas
		h1, w1 = img1.shape[:2]
		h2, w2 = img2.shape[:2]
		result = np.zeros((max(h1, h2 + abs(dy)), w1 + w2 + abs(dx), 3), dtype=np.uint8)

		# Paste first image
		result[:h1, :w1] = img1

		# Paste second image with translation
		y_offset = max(0, dy)
		x_offset = w1 + max(0, dx)
		result[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2

		return result

	stitched = stitch_translation(left, right)

	# stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
	# status, stitched = stitcher.stitch([cv2.imread(image) for image in images])

	print("stitching took", time.time() - start)
	cv2.imwrite('stitched.png', stitched)
	exit()

	location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')

	print('posting to ' + location)

	success, image = cv2.imencode('.png', stitched)

	post(location, data=image.tobytes())
	print('posted')
