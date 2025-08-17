from input_image import InputImage

class SliceInput:
	def __init__(self, image: InputImage):
		self.image = image

		self.offset_x = image.offset_x
