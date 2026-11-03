from requests import post
import cv2

upload_location = 'https://kalkbreite.com/capture'

def upload(canvas):
	success, image = cv2.imencode('.png', canvas)
	print('[upload] posting to ' + upload_location + ', size = ' + str(len(image)))

	post(upload_location, data=image.tobytes())
	print('[upload] complete')
