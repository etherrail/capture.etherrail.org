import cv2
import numpy as np
from input_image import InputImage

def calculate_movement(start: InputImage, end: InputImage, coarse_window):
	# Detect ORB features and descriptors
	orb = cv2.ORB_create(500)
	kp1, des1 = orb.detectAndCompute(start.rotated, None)
	kp2, des2 = orb.detectAndCompute(end.rotated, None)

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

	return -dx
