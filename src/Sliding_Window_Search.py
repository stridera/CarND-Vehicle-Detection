# Sliding Window Implementation

import numpy as np
import cv2

class Sliding_Window_Search():
	def __init__(self, window_size_min=(64, 64), window_size_max=(384, 384), window_size_steps=(64, 64),
		search_window=None, window_overlap=(0.25, 0.25), validator=None, on_valid=None, on_invalid=None, extra_params=None):

		self.window_size_min = window_size_min
		self.window_size_max = window_size_max
		self.window_size_steps = window_size_steps
		self.search_window = search_window
		self.window_overlap = window_overlap
		self.validator = validator
		self.on_valid = on_valid
		self.on_invalid = on_invalid
		self.window = 0
		self.extra_params = extra_params

	def set_search_window(self, search_window):
		self.search_window = search_window

	def process(self, image):
		current_window_size = self.window_size_max
		min_x, min_y = self.search_window[0]
		max_x, max_y = self.search_window[1]
		search_window_width = int(max_x - min_x)
		search_window_height = int(max_y - min_y)

		while current_window_size[0] > self.window_size_min[0] and current_window_size[1] > self.window_size_min[1]:

			x_windows = max(1, np.int((search_window_width - current_window_size[0]) / (current_window_size[0] * self.window_overlap[0])))
			y_windows = max(1, np.int((search_window_height - current_window_size[1]) / (current_window_size[1] * self.window_overlap[1])))

			x_pixels_per_step = np.int((search_window_width - current_window_size[0]) / x_windows)
			y_slide = max(search_window_height - current_window_size[1], 0) 
			# If our window is less than the search window, add one extra search window so we can search the top and bottom
			if y_slide > 0: 
				y_windows += 1

			y_pixels_per_step = np.int(y_slide / y_windows)

			for x_step in range(int(x_windows) + 1):
				for y_step in range(y_windows):
					x = int(x_pixels_per_step * x_step + min_x)
					y = int(max_y - y_pixels_per_step * y_step)
					image = self.check_window(image, (x, y), (x+current_window_size[0], y-current_window_size[1]))
			current_window_size = np.subtract(current_window_size, self.window_size_steps)

		return image

	def check_window(self, image, top_left, bottom_right):
		if (image is None):
			print "Error, no image sent!"
			return
		if (self.validator is not None):
			x1, y2 = top_left
			x2, y1 = bottom_right
			cropped_image = image[y1:y2, x1:x2]
			# print(x1, x2, y1, y2, cropped_image.shape, image.shape)
			if self.validator(cropped_image):
				if self.on_valid is not None:
					self.window += 1
					i = self.window
					image =  self.on_valid(image, cropped_image, top_left, bottom_right, **self.extra_params)
			else:
				if self.on_invalid is not None:
					self.on_invalid(image, cropped_image, top_left, bottom_right, **self.extra_params)
		else:
			print("No Validator")
			
		return image

# Tests

def draw_squares(image, cropped, top_left, bottom_right, color=(0, 0, 255), 
	thickness=5, frame=0, visualize=True, save_frame=False):
	if save_frame:
		cp = np.copy(image)
		cv2.rectangle(cp, top_left, bottom_right, color, thickness)
		r,g,b = cv2.split(cp)
		bgrImage = cv2.merge([b,g,r])
		cv2.imwrite('../output_images/blah{}.png'.format(str(frame).zfill(3)), bgrImage)
	if visualize:
		cv2.rectangle(image, top_left, bottom_right, color, thickness)

	return image

if __name__ == '__main__':
	import glob		
	from matplotlib import pyplot

	# Process
	images = []
	for imgPath in glob.glob('../test_images/*'):
		images.append(imgPath)

	for i, imgPath in enumerate(images):
		image = cv2.imread(imgPath)
		b,g,r = cv2.split(image)
		rgb_image = cv2.merge([r,g,b])

		# Setup 
		sws = Sliding_Window_Search(
			search_window=((0, rgb_image.shape[0] * 0.45), 
				(rgb_image.shape[1], rgb_image.shape[0] * 0.95)),
			validator=lambda image: True,
			on_valid=draw_squares,
			extra_params={'visualize':True, 'save_frame':False}
		)
		processed_image = sws.process(rgb_image)

		pyplot.imshow(processed_image)
		pyplot.show()
		break
