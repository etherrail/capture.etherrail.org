from os import read
from capture import capture
from stitch import stitch
from requests import post
import cv2

while True:
	tag = input('TAG: ')
	direction = input('R/F: ')

	images = capture()
	stitched = stitch(images)

	location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')

	print('posting to ' + location)
	success, image = cv2.imencode('.png', stitched)

	post(location, data=image.tobytes())
	print('posted')
