from os import mkdir
import sys
import tty
import termios
from sys import stdin
import mvsdk
from time import sleep, time
import cv2
import numpy as np
from input_image import InputImage
from stitch import Stitcher
from uuid import uuid4
from subprocess import Popen, PIPE

class Capture(object):
	session = str(uuid4())

	def __init__(self):
		super(Capture, self).__init__()

		mkdir('input/' + self.session)

		self.frameIndex = 0
		self.exposure = 20
		self.frameBuffer = None

		self.stitcher = Popen(
			[sys.executable, 'index.py'],
			stdin=PIPE, stdout=sys.stdout, stderr=sys.stderr,
			text=True
		)

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

		mvsdk.CameraImageProcess(handle, rawData, self.frameBuffer, frameHead)
		mvsdk.CameraReleaseImageBuffer(handle, rawData)

		# prepare file for stitcher, save locally and send next file instruction
		path = 'input/' + self.session + '/' + str(self.frameIndex) + '.png'
		mvsdk.CameraSaveImage(handle, path, self.frameBuffer, frameHead, mvsdk.FILE_BMP, 1000)

		self.stitcher.stdin.write(path + '\n')
		self.stitcher.stdin.flush()

		if self.frameIndex % 20 == 0:
			self.stitcher.stdin.write('NEXT\n')
			self.stitcher.stdin.flush()


	def stop(self):
		mvsdk.CameraUnInit(self.handle)
		mvsdk.CameraAlignFree(self.frameBuffer)

		self.stitcher.stdin.write('FINISH\n')
		self.stitcher.stdin.flush()

		self.stitcher.wait()

		return


if __name__ == '__main__':
	capture = Capture()
	capture.connect()

	capture.start()

	input('* Press ENTER to stop *')

	capture.stop()
