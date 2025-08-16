from os import listdir, read
from sys import stdin
from stitch import stitch
from requests import post
import cv2
import numpy as np
import time

while True:
	tag = input('TAG: ')
	direction = input('R/F: ')

	images = []

	for file in listdir('input'):
		if file.endswith('.bmp'):
			images.append('input/' + file)

	start = time.time()
	stitched = stitch(images)

	print("stitching took", time.time() - start)
	cv2.imwrite('stitched.png', stitched)

	location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')

	print('posting to ' + location)

	success, image = cv2.imencode('.png', stitched)

	post(location, data=image.tobytes())
	print('posted')
