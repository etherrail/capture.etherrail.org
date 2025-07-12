from os import rmdir
from sys import stdin
import mvsdk
from time import sleep

def capture():
	frameIndex = 0
	exposure = 20

	frameBuffer = None
	images = []

	devices = mvsdk.CameraEnumerateDevice()

	if len(devices) < 1:
		print("No camera was found!")

		exit(1)

	camera = devices[0]
	print(camera)

	handle = 0

	try:
		handle = mvsdk.CameraInit(camera, -1, -1)
	except mvsdk.CameraException as e:
		print("camera could not initialize: {} {}".format(e.error_code, e.message) )

		exit(2)

	cap = mvsdk.CameraGetCapability(handle)
	mvsdk.CameraSetIspOutFormat(handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
	mvsdk.CameraSetTriggerMode(handle, 0)
	mvsdk.CameraSetAeState(handle, 0)
	mvsdk.CameraSetExposureTime(handle, exposure * 1000)
	mvsdk.CameraPlay(handle)

	frameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3
	frameBuffer = mvsdk.CameraAlignMalloc(frameBufferSize, 16)

	def capture_callback(handle, rawData, frameHeads, context):
		frameHead = frameHeads[0]
		frameIndex += 1

		mvsdk.CameraImageProcess(handle, rawData, frameBuffer, frameHead)
		mvsdk.CameraReleaseImageBuffer(handle, rawData)

		path = 'input/frame-' + str(frameIndex) + '.bmp'
		images.append(path)

		status = mvsdk.CameraSaveImage(handle, path, frameBuffer, frameHead, mvsdk.FILE_BMP, 1000)

		if status != mvsdk.CAMERA_STATUS_SUCCESS:
			print("Save image failed. {}".format(status))

			exit(3)

	mvsdk.CameraSetCallbackFunction(handle, capture_callback, 0)

	while True:
		if stdin.read(1) == 'e':
			mvsdk.CameraUnInit(handle)
			mvsdk.CameraAlignFree(frameBuffer)

			return images

		sleep(0.1)
