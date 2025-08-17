from os import listdir, read
from sys import stdin
from input_image import InputImage
from stitch import Stitcher
from requests import post
import cv2
import numpy as np
import time

if __name__ == '__main__':
	# tag = input('TAG: ')
	# direction = input('R/F: ')

	stitcher = Stitcher()

	files = []

	for file in listdir('input'):
		if file.endswith('.bmp'):
			files.append('input/' + file)

	files.sort()

	for file in files:
		stitcher.add(InputImage(cv2.imread(file)))

	exit(1)

	location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')

	print('posting to ' + location)

	success, image = cv2.imencode('.png', stitched)

	post(location, data=image.tobytes())
	print('posted')
