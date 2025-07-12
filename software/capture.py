from os import rmdir
from sys import stdin
import mvsdk
from time import sleep

class Capture(object):
	def __init__(self):
		super(Capture, self).__init__()

		self.frameIndex = 0
		self.exposure = 20

		self.frameBuffer = None

	def connect(self):
		devices = mvsdk.CameraEnumerateDevice()

		if len(devices) < 1:
			print("No camera was found!")

			exit(1)

		self.camera = devices[0]
		print(self.camera)

		self.handle = None

	def start(self):
		try:
			self.handle = mvsdk.CameraInit(self.camera, -1, -1)
		except mvsdk.CameraException as e:
			print("camera could not initialize: {} {}".format(e.error_code, e.message) )

			exit(2)

		self.images = []
		self.frameIndex = 0

		capture = mvsdk.CameraGetCapability(self.handle)
		mvsdk.CameraSetIspOutFormat(self.handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
		mvsdk.CameraSetTriggerMode(self.handle, 0)
		mvsdk.CameraSetAeState(self.handle, 0)
		mvsdk.CameraSetExposureTime(self.handle, self.exposure * 1000)
		mvsdk.CameraPlay(self.handle)

		self.frameBufferSize = capture.sResolutionRange.iWidthMax * capture.sResolutionRange.iHeightMax * 3
		self.frameBuffer = mvsdk.CameraAlignMalloc(self.frameBufferSize, 16)

		mvsdk.CameraSetCallbackFunction(self.handle, self.capture_callback, 0)

	@mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
	def capture_callback(self, handle, rawData, frameHeads, context):
		frameHead = frameHeads[0]
		self.frameIndex += 1

		mvsdk.CameraImageProcess(handle, rawData, self.frameBuffer, frameHead)
		mvsdk.CameraReleaseImageBuffer(handle, rawData)

		path = 'input/frame-' + str(self.frameIndex) + '.bmp'
		self.images.append(path)

		status = mvsdk.CameraSaveImage(handle, path, self.frameBuffer, frameHead, mvsdk.FILE_BMP, 1000)

		if status != mvsdk.CAMERA_STATUS_SUCCESS:
			print("Save image failed. {}".format(status))

			exit(3)

		return

	def stop(self):
		mvsdk.CameraUnInit(self.handle)
		mvsdk.CameraAlignFree(self.frameBuffer)

		return self.images
