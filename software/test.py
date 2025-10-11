from os import listdir
from input_image import InputImage
from stitch import Stitcher
from requests import post
import cv2
import re

tag = input('TAG: ')
direction = 'reverse' if input('DIRECTION (f/r): ').lower() == 'r' else 'forward'

stitcher = Stitcher()

frames = [int(re.search(r"\d+", file).group()) for file in listdir('input') if file.endswith('.bmp')]
frames.sort()

for frame in frames:
		image = InputImage(cv2.imread('input/frame-' + str(frame) + '.bmp'))

		# if image.valid_flash_brightness(25, 5, 170, 200):
		stitcher.add(image)

stitched = stitcher.render('')

location = 'https://kalkbreite.com/capture/' + tag + '/' + ('reverse' if direction == 'r' else 'forward')
print('posting to ' + location)

success, image = cv2.imencode('.png', stitched)
post(location, data=image.tobytes())
print('posted')
