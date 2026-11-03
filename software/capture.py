from os import makedirs, path
from datetime import datetime
from queue import Queue
from threading import Thread, Lock
from select import select
import sys
import tty
import termios
import traceback
import mvsdk
import cv2
from stitch import Stitcher
from upload import upload

# one pass of a train
# frames are saved and analyzed while they come in (pass 1 of the stitcher)
# when the session is stopped, the train is rendered and uploaded in the background
class Session:
	def __init__(self, uploading):
		self.name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
		self.directory = path.join('input', self.name)
		self.uploading = uploading

		makedirs(self.directory)

		self.frame_count = 0
		self.analyzed_count = 0
		self.done = False

		self.stitcher = Stitcher()
		self.queue = Queue()

		self.worker = Thread(target=self.work, daemon=True)
		self.worker.start()

	def next_path(self):
		self.frame_count += 1

		return path.join(self.directory, 'frame-' + str(self.frame_count) + '.bmp')

	def add(self, file):
		self.queue.put(file)

	def finish(self):
		self.queue.put(None)

	def work(self):
		try:
			while True:
				file = self.queue.get()

				if file is None:
					break

				self.stitcher.add(file)
				self.analyzed_count += 1

			if self.analyzed_count < 2:
				print('[session] ' + self.name + ' has not enough frames, nothing to stitch')

				return

			print('[session] ' + self.name + ' rendering ' + str(self.analyzed_count) + ' frames')
			canvas = self.stitcher.render()

			output = path.join(self.directory, 'stitched.png')
			cv2.imwrite(output, canvas)
			print('[session] ' + self.name + ' saved to ' + output)

			if self.uploading:
				upload(canvas)
		except Exception:
			print('[session] ' + self.name + ' failed')
			traceback.print_exc()
		finally:
			self.done = True

class Capture:
	exposure = 20 # ms
	exposure_step = 1 # ms

	def __init__(self):
		self.session = None
		self.sessions = []
		self.uploading = True

		# held while a frame is saved, so a session is never stopped in the middle of a frame
		self.lock = Lock()

	def connect(self):
		devices = mvsdk.CameraEnumerateDevice()

		if len(devices) < 1:
			print('[camera] no camera was found!')

			exit(1)

		self.camera = devices[0]
		print('[camera] ' + self.camera.GetFriendlyName())

		try:
			self.handle = mvsdk.CameraInit(self.camera, -1, -1)
		except mvsdk.CameraException as e:
			print('[camera] could not initialize: {} {}'.format(e.error_code, e.message))

			exit(2)

		capability = mvsdk.CameraGetCapability(self.handle)
		mvsdk.CameraSetIspOutFormat(self.handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
		mvsdk.CameraSetTriggerMode(self.handle, 0)
		mvsdk.CameraSetAeState(self.handle, 0)

		self.exposure_range = mvsdk.CameraGetExposureTimeRange(self.handle)
		self.set_exposure(self.exposure)

		self.frame_buffer_size = capability.sResolutionRange.iWidthMax * capability.sResolutionRange.iHeightMax * 3
		self.frame_buffer = mvsdk.CameraAlignMalloc(self.frame_buffer_size, 16)

		mvsdk.CameraSetCallbackFunction(self.handle, self.capture_callback, 0)

		# the camera only runs (and flashes) while a session is recording
		mvsdk.CameraPlay(self.handle)
		mvsdk.CameraPause(self.handle)

	def disconnect(self):
		mvsdk.CameraUnInit(self.handle)
		mvsdk.CameraAlignFree(self.frame_buffer)

	def set_exposure(self, exposure):
		minimum, maximum, step = self.exposure_range
		self.exposure = min(max(exposure * 1000, minimum), maximum) / 1000

		mvsdk.CameraSetExposureTime(self.handle, self.exposure * 1000)
		print('[camera] exposure = ' + str(round(mvsdk.CameraGetExposureTime(self.handle) / 1000, 2)) + ' ms')

	@mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
	def capture_callback(self, handle, raw_data, frame_heads, context):
		frame_head = frame_heads[0]

		with self.lock:
			session = self.session

			if session is None:
				mvsdk.CameraReleaseImageBuffer(handle, raw_data)

				return

			mvsdk.CameraImageProcess(handle, raw_data, self.frame_buffer, frame_head)
			mvsdk.CameraReleaseImageBuffer(handle, raw_data)

			file = session.next_path()
			status = mvsdk.CameraSaveImage(handle, file, self.frame_buffer, frame_head, mvsdk.FILE_BMP, 100)

			if status != mvsdk.CAMERA_STATUS_SUCCESS:
				print('[camera] saving ' + file + ' failed, status = ' + str(status))

				return

			session.add(file)

	def start(self):
		if self.session is not None:
			return

		self.session = Session(self.uploading)
		self.sessions.append(self.session)

		mvsdk.CameraPlay(self.handle)
		print('[session] recording ' + self.session.directory)

	def stop(self):
		if self.session is None:
			return

		mvsdk.CameraPause(self.handle)

		with self.lock:
			session = self.session
			self.session = None

		print('[session] stopped ' + session.name + ' after ' + str(session.frame_count) + ' frames')
		session.finish()

	def toggle(self):
		if self.session is None:
			self.start()
		else:
			self.stop()

	def toggle_upload(self):
		self.uploading = not self.uploading

		# also applies to a session that is still recording
		if self.session is not None:
			self.session.uploading = self.uploading

		print('[upload] ' + ('enabled' if self.uploading else 'disabled'))

	def status(self):
		print('[status] exposure = ' + str(self.exposure) + ' ms, upload ' + ('enabled' if self.uploading else 'disabled'))

		if self.session is not None:
			print('[status] recording ' + self.session.name + ', ' + str(self.session.frame_count) + ' frames, ' + str(self.session.analyzed_count) + ' analyzed')
		else:
			print('[status] idle')

		for session in self.sessions:
			if session is not self.session and not session.done:
				print('[status] stitching ' + session.name + ', ' + str(session.analyzed_count) + ' / ' + str(session.frame_count) + ' frames analyzed')

	def wait(self):
		for session in self.sessions:
			if not session.done:
				print('[session] waiting for ' + session.name + ' to finish')
				session.worker.join()

def help():
	print('')
	print('  space / enter   start or stop recording')
	print('  + / -           exposure up / down')
	print('  u               toggle upload')
	print('  s               status')
	print('  h               help')
	print('  q               quit (finishes stitching first)')
	print('')

if __name__ == '__main__':
	capture = Capture()
	capture.connect()

	help()

	settings = termios.tcgetattr(sys.stdin)
	tty.setcbreak(sys.stdin.fileno())

	try:
		while True:
			ready, _, _ = select([sys.stdin], [], [], 0.5)

			if not ready:
				continue

			key = sys.stdin.read(1)

			if key in (' ', '\n', '\r'):
				capture.toggle()
			elif key in ('+', '='):
				capture.set_exposure(capture.exposure + capture.exposure_step)
			elif key == '-':
				capture.set_exposure(capture.exposure - capture.exposure_step)
			elif key == 'u':
				capture.toggle_upload()
			elif key == 's':
				capture.status()
			elif key in ('h', '?'):
				help()
			elif key == 'q':
				break
	except KeyboardInterrupt:
		pass
	finally:
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

		capture.stop()
		capture.wait()
		capture.disconnect()
