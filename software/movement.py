import cv2
import numpy as np

def calculate_movement(gray1, gray2, coarse_window):
	rows, cols = gray1.shape

	min_diff = 99999999999
	min_movement = 0

	def diff_image(movement):
		dx = movement
		dy = 0

		M = np.float32([[1, 0, dx], [0, 1, dy]])

		translated_image2 = cv2.warpAffine(gray2, M, (cols, rows))

		# Define the valid overlapping region
		x_start = max(0, -dx)
		x_end = min(cols, cols - dx)
		y_start = max(0, dy)
		y_end = min(rows, rows - dy)

		# Extract only the valid region
		roi_image1 = gray1[y_start:y_end, x_start:x_end]
		roi_translated_image2 = translated_image2[y_start:y_end, x_start:x_end]

		# Compute absolute difference within the valid region
		diff = cv2.absdiff(roi_image1, roi_translated_image2)

		# Convert difference to a numerical value (sum of absolute differences)
		sum = np.sum(diff)

		# consider that the image is constantly getting smaller, thus the sum of difference shrinks too
		return sum / diff.size

	# coarse search
	for movement in range(0, int(cols / 2), coarse_window):
		difference_value = diff_image(movement)

		if difference_value < min_diff:
			min_diff = difference_value
			min_movement = movement

	min_coarse = min_movement

	for movement in range(min_coarse - coarse_window, min_coarse + coarse_window):
		difference_value = diff_image(movement)

		if difference_value < min_diff:
			min_diff = difference_value
			min_movement = movement

	return min_movement
