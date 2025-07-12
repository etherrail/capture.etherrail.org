from os import read
from sys import stdin
from capture import Capture
from stitch import stitch
from requests import post
import cv2

capture = Capture()
capture.connect()

while True:
	tag = input('TAG: ')
	direction = input('R/F: ')

	capture.start()
	images = []

	while True:
		if stdin.read(1) == 'e':
			images = capture.stop()

			break

	stitched = stitch(images)

	location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')

	print('posting to ' + location)
	success, image = cv2.imencode('.png', stitched)

	post(location, data=image.tobytes())
	print('posted')
