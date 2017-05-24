import cv2
import numpy as np
import os.path
from moviepy.editor import VideoFileClip

from Sliding_Window_Search import Sliding_Window_Search
from Image_Utils import Image_Utils
from Classifier import Classifier

from matplotlib import pyplot

class CarFindingPipeline():
	def __init__(self, train_classifier=False):
		self.iu = Image_Utils()
		self.classifier = Classifier(train_classifier)
		self.sws = Sliding_Window_Search(
			validator=self.process_image,
			on_valid=self.draw_squares,
			extra_params={'visualize':True, 'save_frame':False}
		)

	def update_search_window(self, search_window):
		self.sws.set_search_window(search_window)

	def draw_squares(self, image, top_left, bottom_right, color=(0, 0, 255), 
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

	def process_image(self, frame):
		features = self.iu.extract_features_from_image(frame)
		results = self.classifier.predict([features])

		if results[0] == 1:	
			print "CAR!"
		# 	title = "Car"
		# else:
		# 	title = "No Car"
		# pyplot.imshow(frame)
		# pyplot.title(title)
		# pyplot.show()
		
		return results[0] == 1

	def process(self, frame):
		self.sws.process(frame)
		return frame


def processVideo(path):
	pipeline = LaneFindingPipeline(False)

	filename, file_extension = os.path.splitext(path)
	challenge_output_path = "{}-processed{}".format(filename, file_extension)
	clip = VideoFileClip(path)
	processed_clip = clip.fl_image(pipeline.process)
	processed_clip.write_videofile(challenge_output_path, audio=False)

def processTestImages():
	import glob
	from matplotlib import pyplot

	images = []

	images = glob.glob('../test_images/*')

	pipeline = CarFindingPipeline(True)

	cols = len(images)
	for i, imgPath in enumerate(images):
		image = cv2.imread(imgPath)
		b,g,r = cv2.split(image)
		rgb_image = cv2.merge([r,g,b])

		search_window = (
			(0, rgb_image.shape[0] * 0.45), 
			(rgb_image.shape[1], rgb_image.shape[0] * 0.95))

		pipeline.update_search_window(search_window)
		processedImage = pipeline.process(rgb_image)

		# r,g,b = cv2.split(processedImage)
		# bgrImage = cv2.merge([b,g,r])
		# cv2.imwrite('../output_images/' + os.path.basename(imgPath), bgrImage)

		pyplot.subplot(121)
		pyplot.axis('off')
		pyplot.title(imgPath)
		pyplot.imshow(rgb_image)
		pyplot.subplot(122)
		pyplot.title("{}, {}".format(cols*2, (i*2)))
		pyplot.imshow(processedImage)
	
		pyplot.tight_layout()
		pyplot.show()

		# break


def main():
	''' Main Function '''
	processTestImages()
	# processVideo("../project_video.mp4")
	# processVideo("../test_video.mp4")





if __name__ == '__main__':
	main()

