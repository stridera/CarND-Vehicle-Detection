import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


class Image_Utils():
	def extract_features(self, images, cspace='YCrCb', hog_channel=None):
		# Create a list to append feature vectors to
		features = []
		# Iterate through the list of images
		for file in images:
			# Read in each one by one
			image = mpimg.imread(file)

			# Resize Image
			resized = cv2.resize(image, (64, 64))

			# apply color conversion if other than 'RGB'
			color_converted_image = self.convert_color_space(resized, cspace)
			
			# Get hog features
			hog_features = self.get_image_hog_feature(color_converted_image, hog_channel)
			
			# Append the new feature vector to the features list
			features.append(hog_features)
			
		# Return list of feature vectors
		return features

	def extract_features_from_image(self, image, cspace='RGB', hog_channel=None):
		resized = cv2.resize(image, (64, 64))		
		color_converted_image = self.convert_color_space(resized, cspace)
		return self.get_image_hog_feature(color_converted_image, hog_channel)

	def convert_color_space(self, image, cspace):
		# apply color conversion if other than 'RGB'
		if cspace != 'RGB':
			if cspace == 'HSV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
		return image

	def get_image_hog_feature(self, image, hog_channel=None):
		# Call get_hog_features() with vis=False, feature_vec=True
		if hog_channel is None:
			hog_features = []
			for channel in range(image.shape[2]):
				hog_features.append(self.get_hog_features(image[:,:,channel], visualise=False))
			return np.ravel(hog_features)        
		else:
			return self.get_hog_features(image[:,:,hog_channel], visualise=False)

	# Define a function to return HOG features and visualization
	def get_hog_features(self, image, orient=9, pix_per_cell=8, cell_per_block=3, visualise=False, feature_vec=True):
		if visualise == True:
			features, hog_image = hog(
				image, 
				orientations=orient, 
				pixels_per_cell=(pix_per_cell, pix_per_cell),
				cells_per_block=(cell_per_block, cell_per_block), 
				transform_sqrt=False, 
				visualise=True, 
				block_norm='L2-Hys',
				feature_vector=False)
			return features, hog_image
		else:      
			features = hog(image, 
				orientations=orient, 
				pixels_per_cell=(pix_per_cell, pix_per_cell),
				cells_per_block=(cell_per_block, cell_per_block), 
				transform_sqrt=False, 
				visualise=False, 
				block_norm='L2-Hys',
				feature_vector=feature_vec)
			return features

	def bin_spatial(self, img, size=(32, 32)):
		color1 = cv2.resize(img[:,:,0], size).ravel()
		color2 = cv2.resize(img[:,:,1], size).ravel()
		color3 = cv2.resize(img[:,:,2], size).ravel()
		return np.hstack((color1, color2, color3))
							
	def color_hist(self, img, nbins=32):    #bins_range=(0, 256)
		# Compute the histogram of the color channels separately
		channel1_hist = np.histogram(img[:,:,0], bins=nbins)
		channel2_hist = np.histogram(img[:,:,1], bins=nbins)
		channel3_hist = np.histogram(img[:,:,2], bins=nbins)
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return hist_features




if __name__ == '__main__':
	import glob		
	from matplotlib import pyplot

	# Process
	images = glob.glob('../test_images/*')

	image_utils = Image_Utils()
	print image_utils.extract_features(images, 2)
