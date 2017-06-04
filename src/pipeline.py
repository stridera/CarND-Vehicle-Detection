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
			spatial_size=(32, 32), 
			hist_bins=32, 
			orient=9, 
			pix_per_cell=8, 
			cell_per_block=2, 
			feature_vec=True,
			use_spatial=True,
			use_histogram=True
		)
		self.classifier = Classifier(
			train_classifier, 
			image_utils=self.iu
		)
		self.sws = Sliding_Window_Search(
			window_size_steps=(32, 32),
			window_size_min=(64, 64), 
			window_size_max=(128, 128), 
			window_overlap=(0.25, 0.25),
			validator=self.process_image,
			# validator=lambda x: True,
			# on_valid=lambda cropped_image, **kwargs: self.save_image(cropped_image, 'valid', 0.01),
			# on_invalid=lambda cropped_image, **kwargs: self.save_image(cropped_image, 'invalid', 0.01),

			on_valid=self.save_squares,
			# on_invalid=self.save_squares,

			# on_invalid=self.save_image,
			extra_params={'visualize':True, 'save_frame':True}
		)
		self.heatmap_threshold = 2
		self.search_window_set = False
		self.image_number = 0
		self.frame = 0
		self.squares = []
		self.heatmap_queue = deque(maxlen = 15)


	def reset(self):
		self.image_number = 0
		self.frame = 0
		self.squares = []
		self.heatmap_queue = deque(maxlen = 5)	

	def update_search_window(self, search_window):
		self.sws.set_search_window(search_window)
		self.search_window_set = True

	def save_squares(self, image, cropped, top_left, bottom_right, **kwargs):
		self.squares.append((top_left, bottom_right))
		return image

	def save_image(self, image, directory='.', keep_rate=1, **kwargs):
		self.image_number += 1	

		rand = np.random.randint(0, 100)
		if rand < keep_rate:
			return image

		if not os.path.exists(directory):
			os.makedirs(directory)

		bgr_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
		cv2.imwrite('{}frame{}image{}.png'.format(
				directory,
				str(self.frame).zfill(5),
				str(self.image_number).zfill(5)
			),
			bgr_image)

		return image

	def draw_squares(self, image, top_left, bottom_right, color=(0, 0, 255), 
		thickness=5, visualize=True, save_frame=False):
		if save_frame:
			this.save_frame(image)
		if visualize:
			cv2.rectangle(image, top_left, bottom_right, color, thickness)

		return image

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
		self.heatmap_queue.append(heat)#np.clip(heat, 0, 255))
		avg = np.mean(self.heatmap_queue, axis=0)
		avg[avg <= 8 ] = 0
		return avg

	def draw_overlay_using_heatmap_labels(self, image):
		final = np.copy(image)
		heatmap = self.generage_heat_map(image)
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
			w = bbox[1][0] - bbox[0][0]
			h = bbox[1][1] - bbox[0][1]
			if w>50 and h>50:
				self.draw_squares(final, bbox[0], bbox[1])

			# pyplot.subplot(1, labels[1], car_number)
			# pyplot.axis('off')
			# pyplot.imshow(labels[0] == car_number, cmap='hot')
	
		# pyplot.imshow(final)
		# pyplot.axis('off')
		# pyplot.title("Cars Found: {}".format(labels[1]))
		# pyplot.show()

		# Return the image
		return final
	
	def process_image(self, frame):
		''' Used to process the cropped image within the sliding window '''
		self.features = self.iu.process(frame)
		results = self.classifier.predict([self.features])
		return results[0] == 1

	def process(self, frame):
		self.frame += 1
		self.image_number = 0
		self.squares = []

		if not self.search_window_set:
			search_window = (
				(frame.shape[1]*0.55, frame.shape[0] * 0.45), 
				(frame.shape[1], frame.shape[0] * 0.95)
			)
			self.update_search_window(search_window)

		self.sws.process(frame)
		return self.draw_overlay_using_heatmap_labels(frame)

	def get_heatmap(self):
		return np.mean(self.heatmap_queue)

def processVideo(path, train=False):
	pipeline = CarFindingPipeline(train)

	filename, file_extension = os.path.splitext(path)
	challenge_output_path = "{}-processed{}".format(filename, file_extension)
	clip = VideoFileClip(path)#.subclip(20, 45)
	processed_clip = clip.fl_image(pipeline.process)
	processed_clip.write_videofile(challenge_output_path, audio=False)

def processTestImages(train=False):
	import glob
	from matplotlib import pyplot
	import matplotlib.image as mpimg

	images = []

	images = glob.glob('../test_images/*')

	pipeline = CarFindingPipeline(train)

	cols = len(images)
	for i, imgPath in enumerate(images):
		pipeline.reset()

		image = cv2.imread(imgPath)
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# rgb_image = mpimg.imread(imgPath)

		search_window = (
			(rgb_image.shape[1]*0.55, rgb_image.shape[0] * 0.45), 
			(rgb_image.shape[1], rgb_image.shape[0] * 0.95))

		pipeline.update_search_window(search_window)
		processedImage = pipeline.process(rgb_image)
		heatmap = pipeline.get_heatmap()
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

