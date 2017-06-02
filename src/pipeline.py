import cv2
import numpy as np
import os.path
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from collections import deque

from Sliding_Window_Search import Sliding_Window_Search
from Image_Utils import Image_Utils
from Classifier import Classifier

from matplotlib import pyplot

class CarFindingPipeline():
	def __init__(self, train_classifier=False):
		self.iu = Image_Utils(
			cspace='YCrCb', 
			hog_channel=None, 
			spatial_size=(16, 16), 
			hist_bins=128, 
			orient=9, 
			pix_per_cell=8, 
			cell_per_block=2, 
			feature_vec=True,
			use_spatial=False,
			use_histogram=True
		)
		self.classifier = Classifier(train_classifier, image_utils=self.iu)
		self.sws = Sliding_Window_Search(
			window_size_steps=(32, 32),

			validator=self.process_image,
			on_valid=self.save_squares,
			# on_invalid=self.save_squares,

			# on_invalid=self.save_image,
			extra_params={'visualize':True, 'save_frame':True}
		)
		self.search_window_set = False
		self.image_number = 0
		self.squares = []
		self.heatmap_queue = deque(maxlen = 5)


	def reset(self):
		self.image_number = 0
		self.squares = []
		self.heatmap_queue = deque(maxlen = 5)	

	def update_search_window(self, search_window):
		self.sws.set_search_window(search_window)
		self.search_window_set = True

	def save_squares(self, image, cropped, top_left, bottom_right, **kwargs):
		self.squares.append((top_left, bottom_right))
		return image

	def save_image(self, image, cropped_image, top_left, bottom_right):
		self.image_number += 1	
	
		r,g,b = cv2.split(cropped_image)
		bgrImage = cv2.merge([b,g,r])
		cv2.imwrite('../output_images/blah{}.png'.format(str(self.image_number).zfill(3)), bgrImage)

	def draw_squares(self, image, top_left, bottom_right, color=(0, 0, 255), 
	thickness=5, visualize=True, save_frame=False):
		if save_frame:
			this.save_frame(image)
		if visualize:
			cv2.rectangle(image, top_left, bottom_right, color, thickness)

		return image

	def process_image(self, frame):
		# self.features = self.iu.extract_features_from_image(frame)
		self.features = self.iu.process(frame)
		results = self.classifier.predict([self.features])

		# if results[0] > 0:	
		# 	print("CAR!")
		# 	title = "Car"
		# else:
		# 	title = "No Car"
		# pyplot.imshow(frame)
		# pyplot.title(title)
		# pyplot.show()
		
		return results[0] == 1

	def process(self, frame):
		self.reset()
		# frame = self.iu.extract_features_from_image(frame)

		# pyplot.imshow(frame)
		# pyplot.show()
		if not self.search_window_set:
			search_window = (
				(0, frame.shape[0] * 0.45), 
				(frame.shape[1], frame.shape[0] * 0.95)
			)
			self.update_search_window(search_window)

		self.sws.process(frame)

		return self.draw_overlay_using_heatmap_labels(frame)

	def draw_squares_on_image(self, image):
		marked_image = np.copy(image)
		for square in self.squares:
			tl, br = square
			image = self.draw_squares(marked_image, tl, br)
		return marked_image

	def generage_heat_map(self, image):
		heat = np.zeros_like(image[:,:,0]).astype(np.float)
		for square in self.squares:
			tl, br = square
			heat[br[1]:tl[1], tl[0]:br[0]] += 1
		heat[heat <= 50] = 0
		self.heatmap_queue.append(np.clip(heat, 0, 255))
		return np.average(self.heatmap_queue, axis=0)

	def draw_overlay_using_heatmap(self, image):
		final = np.copy(image)
		if self.heatmap_queue is None:
			heatmap = self.generage_heat_map(image)
		
		heatmap = np.average(self.heatmap_queue)

		zeros = np.zeros_like(heatmap)
		heatmap3 = cv2.merge((zeros, zeros, heatmap))

		final = cv2.addWeighted(image, 1.0, np.uint8(heatmap3), 2.0, 0)
		return final

	def draw_overlay_using_heatmap_labels(self, image):
		final = np.copy(image)
		if not self.heatmap_queue:
			self.generage_heat_map(image)
		heatmap = np.average(self.heatmap_queue, axis=0)

		labels = label(heatmap)

		for car_number in range(1, labels[1]+1):
			# Find pixels with each car_number label value
			nonzero = (labels[0] == car_number).nonzero()
			# Identify x and y values of those pixels
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			# Define a bounding box based on min/max x and y
			bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
			# Draw the box on the image
			self.draw_squares(final, bbox[0], bbox[1])
	
		# Return the image
		return final
		
def processVideo(path):
	pipeline = CarFindingPipeline()

	filename, file_extension = os.path.splitext(path)
	challenge_output_path = "{}-processed{}".format(filename, file_extension)
	clip = VideoFileClip(path)
	processed_clip = clip.fl_image(pipeline.process)
	processed_clip.write_videofile(challenge_output_path, audio=False)

def processTestImages():
	import glob
	from matplotlib import pyplot
	import matplotlib.image as mpimg

	images = []

	images = glob.glob('../test_images/*')

	pipeline = CarFindingPipeline(True)

	cols = len(images)
	for i, imgPath in enumerate(images):
		pipeline.reset()

		image = cv2.imread(imgPath)
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# rgb_image = mpimg.imread(imgPath)

		search_window = (
			(0, rgb_image.shape[0] * 0.45), 
			(rgb_image.shape[1], rgb_image.shape[0] * 0.95))

		pipeline.update_search_window(search_window)
		processedImage = pipeline.process(rgb_image)
		heatmap = pipeline.generage_heat_map(rgb_image)
		marked = pipeline.draw_squares_on_image(rgb_image)
		# final = pipeline.draw_overlay_using_heatmap(rgb_image)
		# final = pipeline.draw_overlay_using_heatmap_labels(rgb_image)

		# r,g,b = cv2.split(final)
		# bgr_image = cv2.merge([b,g,r])
		# cv2.imwrite('../output_images/' + os.path.basename(imgPath), bgr_image)
		
		pyplot.subplot(221)
		pyplot.axis('off')
		pyplot.imshow(rgb_image)
		pyplot.subplot(222)
		pyplot.axis('off')
		pyplot.imshow(marked)
		pyplot.subplot(223)
		pyplot.axis('off')
		pyplot.imshow(heatmap, cmap='hot')
		pyplot.subplot(224)
		pyplot.axis('off')
		pyplot.imshow(processedImage)

		pyplot.tight_layout()
		pyplot.show()



def main():
	''' Main Function '''
	# processTestImages()
	processVideo("../project_video.mp4")
	# processVideo("../test_video.mp4")





if __name__ == '__main__':
	main()

