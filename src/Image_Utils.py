import cv2
import numpy as np
import matplotlib.image as mpimage
from skimage.feature import hog
import tqdm



class Image_Utils():
	def __init__(self, cspace='YCrCb', hog_channel=None, spatial_size=(32, 32), hist_bins=32, 
			orient=9, pix_per_cell=8, cell_per_block=3, feature_vec=True, use_spatial=True, use_histogram=True):
		self.cspace = cspace
		self.hog_channel = hog_channel
		self.spatial_size = spatial_size
		self.hist_bins = hist_bins
		self.orient=orient
		self.pix_per_cell=pix_per_cell
		self.cell_per_block=cell_per_block
		self.feature_vec=feature_vec
		self.use_spatial = use_spatial
		self.use_histogram = use_histogram


	def process(self, images):
		if isinstance(images, list):
			return self.extract_features_from_image_path_list(images)
		else:
			return self.extract_features_from_image(images)

	def extract_features_from_image_path_list(self, path_list):
		# Create a list to append feature vectors to
		features = []
		# Iterate through the list of images
		for image_path in tqdm.tqdm(path_list):
			# Read in each one by one
			# image = mpimage.imread(image_path)
			image = cv2.imread(image_path)
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# Get features
			image_features = self.extract_features_from_image(rgb_image)
			
			# Append the new feature vector to the features list
			features.append(image_features)
			
		# Return list of feature vectors
		return features

	def extract_features_from_image(self, image):
		resized = cv2.resize(image, (64, 64))		
		color_converted_image = self.convert_color_space(resized)
		return self.get_image_features(color_converted_image)

	def get_image_features(self, image):
		features = []
		features.append(self.get_image_hog_feature(image))
		if self.use_spatial:
			features.append(self.get_bin_spatial_features(image))
		if self.use_histogram:
			features.append(self.get_histogram_features(image))

		return np.concatenate(features, axis=0)

	def convert_color_space(self, image):
		# apply color conversion if other than 'RGB'
		if self.cspace != 'RGB':
			if self.cspace == 'HSV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif self.cspace == 'LUV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif self.cspace == 'HLS':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif self.cspace == 'YUV':
				image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif self.cspace == 'YCrCb':
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
		return image

	def get_image_hog_feature(self, image, visualise=False):
		# Call get_hog_features() with vis=False, feature_vec=True
		if self.hog_channel is None:
			hog_features = []
			for channel in range(image.shape[2]):
				hog_features.append(self.get_hog_features(image[:,:,channel], visualise=visualise))
			return np.ravel(hog_features)        
		else:
			return self.get_hog_features(image[:,:,self.hog_channel], visualise=visualise)

	# Define a function to return HOG features and visualization
	def get_hog_features(self, image, visualise=False):
		if visualise == True:
			features, hog_image = hog(
				image, 
				orientations=self.orient, 
				pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
				cells_per_block=(self.cell_per_block, self.cell_per_block), 
				transform_sqrt=True, 
				visualise=True, 
				block_norm='L2-Hys',
				feature_vector=self.feature_vec)
			return features, hog_image
		else:      
			features = hog(image, 
				orientations=self.orient, 
				pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
				cells_per_block=(self.cell_per_block, self.cell_per_block), 
				transform_sqrt=True, 
				visualise=False, 
				block_norm='L2-Hys',
				feature_vector=self.feature_vec)
			return features

	def get_bin_spatial_features(self, image):
		color1 = cv2.resize(image[:,:,0], self.spatial_size).ravel()
		color2 = cv2.resize(image[:,:,1], self.spatial_size).ravel()
		color3 = cv2.resize(image[:,:,2], self.spatial_size).ravel()
		return np.hstack((color1, color2, color3))
							
	def get_histogram_features(self, image):    #bins_range=(0, 256)
		# Compute the histogram of the color channels separately
		channel1_hist = np.histogram(image[:,:,0], bins=self.hist_bins)
		channel2_hist = np.histogram(image[:,:,1], bins=self.hist_bins)
		channel3_hist = np.histogram(image[:,:,2], bins=self.hist_bins)
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return hist_features




if __name__ == '__main__':
	import glob		
	from matplotlib import pyplot

	# Process
	vehicles_path = '../data/vehicles/**/*.png'	
	images = glob.glob(vehicles_path)
	image = cv2.imread(images[0])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image_utils = Image_Utils(hog_channel=0)
	# print(image_utils.process(images))
	features, hog_image = image_utils.get_image_hog_feature(image, True)
	pyplot.imshow(hog_image, cmap='gray')
	pyplot.show()
