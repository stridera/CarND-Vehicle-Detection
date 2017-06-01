import cv2
import numpy as np
import matplotlib.image as mpimg
import skimage
import tqdm

class Hog(object):
	"""Hog Feature class

	Attributes:
		params (dict): parameters for `skimage.feature.hog`

	Examples:
		>>> hog = Hog()
		>>> hog.apply(image, hog_channel=0)

	Keyword Args:
		orientations (int): Orientations of the gradients (default: 9)
		
		pixels_per_cell (tuple): One cell is equal to ROW x COL (default: (8, 8))
			
		cells_per_block (tuple): One block consists of ROW x COL cells (default: (3, 3))            
			
		block_norm (str): {"L1", "L2", "L2-Hys"} (default: "L2-Hys)
		
		visualise (bool): Returns an image (default: False)
		
		transform_sqrt (bool): Apply power law compression to normalize (default: False)
		
		feature_vector (bool): Returns a feature vector (default: True)

	"""

	def __init__(self,
				 orientations=9,
				 pixels_per_cell=(8, 8),
				 cells_per_block=(3, 3),
				 block_norm='L2-Hys',
				 visualise=False,
				 transform_sqrt=False,
				 feature_vector=True):

		self.params = {
			"orientations": orientations,
			"pixels_per_cell": pixels_per_cell,
			"cells_per_block": cells_per_block,
			"block_norm": block_norm,
			"visualise": visualise,
			"transform_sqrt": transform_sqrt,
			"feature_vector": feature_vector,
		}

	def apply(self, image, hog_channel):
		"""Returns hog features from the image
		
		Args:
			image (3-D array): image in numpy array (H, W, C)
			hog_channel (int): channel of the image to extract hog features
			
		Returns:
			hog_features (1-D array): when `feature_vector` is set to True
			hog_image (2-D array, optional): when `visualise` is True
		"""
		H, W, C = image.shape  
		
		if type(hog_channel) == str and hog_channel == "ALL":
			features = []
			for c in range(C):
				partial = image[..., c]
				feat = skimage.feature.hog(partial, **self.params)
				features.append(feat)
			
			return np.concatenate(features, axis=0)
		
		elif type(hog_channel) == int:
			assert hog_channel < C, "Image shape: {} Requested hog channel: {}".format(image.shape, hog_channel)
			
			partial = image[..., hog_channel]
			return skimage.feature.hog(partial, **self.params)


class Image_Utils2():
	def read_image(self, path):
		"""Returns a RGB image"""
		image = cv2.imread(path)
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	def extract_feature_from_single_image(self, image,
									  color_space='RGB',
									  spatial_size=(32, 32),
									  hist_bins=32,
									  hog=None, hog_channel=0,
									  spatial_feat=True, hist_feat=True, hog_feat=True):
		"""Extract features from single image

			1) Change Color space
			2) Get spatial features
			3) Get histogram features
			4) Get hog features

		Args:
			image (3-D array): RGB image array shape (H, W, C)
			color_space (str): choose one {"RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"}
			spatial_size (tuple): image size for spatial_features
			hist_bins (int): histogram bins
			hog_channel (int or str): channel (int) or "ALL" for all
			spatial_feat (bool): spatial features array shape (spatial_size[0] * spatial_size[1], )
			hist_feat (bool): shape (32 * n_channel, )
			hog_feat (bool): hog features

		Returns:
			features (1-D array): numpy array shape (n_dimensions, )
		"""
		features = []
		
		image = cv2.resize(image, (64, 64))
		image = self.convert_color_space(image, color_space)
		
		if spatial_feat:
			feat = self.get_bin_spatial_features(image, spatial_size)        
			features.append(feat)
		
		if hist_feat:
			feat = self.get_histogram_features(image, hist_bins)
			features.append(feat)
		
		if hog_feat:    
			if not hog:
				print "Hog not defined"
				return        
			feat = hog.apply(image, hog_channel)
			features.append(feat)
			
		return np.concatenate(features, axis=0)

	def extract_features(self, images,
						 color_space='RGB',
						 spatial_size=(32, 32),
						 hist_bins=32,
						 hog=None, hog_channel=0,
						 spatial_feat=True, hist_feat=True, hog_feat=True):
		"""Extract features from multiple images(path)

			1) Change Color space
			2) Get spatial features
			3) Get histogram features
			4) Get hog features

		Args:
			images (list): ["/path/to/image", "/path/to/another/image", ...]
			color_space (str): choose one {"RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"}
			spatial_size (tuple): image size for spatial_features
			hist_bins (int): histogram bins
			hog_channel (int or str): channel (int) or "ALL" for all
			spatial_feat (bool): spatial features array shape (spatial_size[0] * spatial_size[1], )
			hist_feat (bool): shape (32 * n_channel, )
			hog_feat (bool): hog features

		Returns:
			features (2-D array): numpy array shape (n_samples, n_dimensions)
		"""
		features = []
		
		for image in tqdm.tqdm(images):
			image = self.read_image(image)        
			feat = self.extract_feature_from_single_image(image, color_space, spatial_size, hist_bins, hog, hog_channel, spatial_feat, hist_feat, hog_feat)
			features.append(feat)
		return np.vstack(features)

	def get_bin_spatial_features(self, image, size=(32, 32)):
		"""Resizes the image and flatten using ravel
		
		Args:
			image (array_like): numpy image 2-D or 3-D array
			size (tuple): resizes the image to (int, int)
		
		Returns:
			1-D array: spatial(flatten) features of the resized image
		"""    
		return cv2.resize(image, size).ravel()

	def get_histogram_features(self, image, bins=32, bin_range=(0, 256)):
		"""Returns histogram features out of the image
		
		Args:
			bins (int): number of histogram bins for `np.histogram`
			bin_range (tuple): range of color (default: (0, 256))
			
		Returns:
			1-D array: histogram features for every channel of the image
		"""
		assert len(image.shape) == 3, "{} is not 3-D".format(image.shape)    
		C = image.shape[-1]
		
		features = []
		
		for i in range(C):
			hist, bins = np.histogram(image[..., i], bins, range=bin_range)
			features.append(hist)
		
		return np.concatenate(features, axis=0)

	def convert_color_space(self, image, color_space="RGB"):
		"""Returns an image in given color space
		
		Args:
			image (array): 2-D or 3-D numpy array.
				The color space must be `RGB`
				
			color_space (str): choose one {"RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"}
			
		Returns:
			image (array): same shape as `image` but in different color space
		"""
		
		if color_space == "RGB":
			return np.copy(image)
		elif color_space == "HSV":
			return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		elif color_space == "LUV":
			return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
		elif color_space == "HLS":
			return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		elif color_space == "YUV":
			return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
		elif color_space == "YCrCb":
			return cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
		else:
			warnings.warn("{} space is not supported".format(color_space))
			return np.copy(image)

	def process(self, image):
		hog_params = {
			"orientations": 9,
			"pixels_per_cell": (8, 8),
			"cells_per_block": (2, 2),
			"visualise": False
		}

		hog = Hog(**hog_params)


		params = {
			"color_space": 'HSV',
			"spatial_size": (32, 32),
			"hist_bins": 32,
			"hog": hog,
			"hog_channel": "ALL",
			"spatial_feat": True, 
			"hist_feat": True, 
			"hog_feat": True,
		}

		if isinstance(image, list):
			return self.extract_features(image, **params)
		else:
			return self.extract_feature_from_single_image(image, **params)