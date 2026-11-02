from os import listdir, path
from stitch import Stitcher
from requests import post
import cv2
import re
import sys
from subprocess import check_output

upload_location = 'https://kalkbreite.com/capture'

# usage: index.py [input directory] [--no-upload]
arguments = [argument for argument in sys.argv[1:] if not argument.startswith('--')]
upload = '--no-upload' not in sys.argv
directory = arguments[0] if len(arguments) else 'input'

stitcher = Stitcher()

frames = [int(re.search(r"\d+", file).group()) for file in listdir(directory) if file.endswith('.bmp')]
frames.sort()

print('▄▖▗ ▌     ▄▖  ▘▜   ▄▖    ▗       ')
print('▙▖▜▘▛▌█▌▛▘▙▘▀▌▌▐   ▌ ▀▌▛▌▜▘▌▌▛▘█▌')
print('▙▖▐▖▌▌▙▖▌ ▌▌█▌▌▐▖  ▙▖█▌▙▌▐▖▙▌▌ ▙▖')
print('#' + check_output(['git','rev-parse','HEAD']).decode().strip()[0:7] + '               ▌         ')
print('Capture Stitcher')
print('')
print('input images: ' + ' '.join(str(frame) for frame in frames))
print('')

for frame in frames:
	file = path.join(directory, 'frame-' + str(frame) + '.bmp')
	print('[image] importing ' + file)

	stitcher.add(file)

canvas = stitcher.render()

cv2.imwrite('stitched.png', canvas)

if upload:
	success, image = cv2.imencode('.png', canvas)
	print('[upload] posting to ' + upload_location + ', size = ' + str(len(image)))

	post(upload_location, data=image.tobytes())
	print('[upload] complete')
