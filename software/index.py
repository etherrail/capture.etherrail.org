from os import unlink
from input_image import InputImage
from stitch import Stitcher
import cv2

print('write filename to add file to stitcher. the source image will be deleted after import')
print('write NEXT to generate a frame from all images (and reset position)')
print('write FINISH to generate a frame, then exit')

stitcher = Stitcher()

while True:
	file = input()

	if file == 'NEXT' or file == 'FINISH':
		stitcher.render()

		if file == 'FINISH':
			exit(0)

	image = cv2.imread(file)
	unlink(file)

	stitcher.add(InputImage(image))
