from os import unlink
from input_image import InputImage
from stitch import Stitcher
import cv2
from requests import get

print('write filename to add file to stitcher. the source image will be deleted after import')
print('write NEXT to generate a frame from all images (and reset position)')
print('write FINISH to generate a frame, then exit')

print('obtaining session...')
session = get('https://kalkbreite.com/capture/session/create').text
print('kalkbreite session ' + session)

stitcher = Stitcher()

while True:
	file = input()

	if file == 'NEXT' or file == 'FINISH':
		stitcher.render(session)

		if file == 'FINISH':
			exit(0)
	else:
		image = cv2.imread(file)
		unlink(file)

		stitcher.add(InputImage(image))
