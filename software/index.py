from os import unlink
from input_image import InputImage
from stitch import Stitcher
import cv2

print('write filename to add file to stitcher. the source image will be deleted after import')
print('write NEXT to generate a frame from all images (and reset position)')
print('write FINISH to generate a frame, then exit')
print('a new frame will be rendered after 10k pixels automatically')

stitcher = Stitcher()
session = ''

while True:
	file = input()

	if file == 'NEXT' or file == 'FINISH':
		stitcher.render(session)

		if file == 'FINISH':
			exit(0)
	else:
		image = InputImage(cv2.imread(file))

		if image.valid_flash_brightness(25, 5, 200, 247):
			stitcher.add(image)

			if stitcher.total_movement_x > 10000:
				stitcher.render(session)
		else:
			print('invalid brightness of image: ' + file + ', brightness')
