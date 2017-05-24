import cv2
import numpy as np
import os.path
from moviepy.editor import VideoFileClip

from Sliding_Window_Search import Sliding_Window_Search
from Image_Utils import Image_Utils

class CarFindingPipeline():
	def __init__(self):
		''' blah '''

	def process(self, frame):

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

	cols = len(images)
	for i, imgPath in enumerate(images):
		image = cv2.imread(imgPath)
		b,g,r = cv2.split(image)
		rgbImage = cv2.merge([r,g,b])

		pipeline = CarFindingPipeline(True)
		processedImage = pipeline.process(rgbImage)

		# r,g,b = cv2.split(processedImage)
		# bgrImage = cv2.merge([b,g,r])
		# cv2.imwrite('../output_images/' + os.path.basename(imgPath), bgrImage)

		pyplot.subplot(121)
		pyplot.axis('off')
		pyplot.title(imgPath)
		pyplot.imshow(rgbImage)
		pyplot.subplot(122)
		pyplot.title("{}, {}".format(cols*2, (i*2)))
		pyplot.imshow(processedImage)
	
		pyplot.tight_layout()
		pyplot.show()

		break


def main():
	''' Main Function '''
	processTestImages()
	# processVideo("../project_video.mp4")
	# processVideo("../test_video.mp4")





if __name__ == '__main__':
	main()

