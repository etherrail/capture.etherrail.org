import cv2
import numpy as np
from input_image import InputImage

def calculate_movement(start: InputImage, end: InputImage, coarse_window):
	def calculate_movement_phase_correlation(start_gray, end_gray):
		# Ensure input is float32 for phaseCorrelate
		img1 = np.float32(start_gray)
		img2 = np.float32(end_gray)

		shift, response = cv2.phaseCorrelate(img1, img2)
		return shift

	dx, dy = calculate_movement_phase_correlation(start.edge_mask, end.edge_mask)

	return int(-dx)
