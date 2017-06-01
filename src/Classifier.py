import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from Image_Utils import Image_Utils
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pickle


class Classifier:
	def __init__(self, train_data=False, image_utils=None):
		self.data_path='../data/classifier.pickle'	

		if image_utils is None:
			self.image_utils = Image_Utils()
		else:
			self.image_utils = image_utils

		if train_data:
			self.svc = LinearSVC()
			self.train_classifier()
		else:
			with open(self.data_path, "rb") as f:
				self.svc = pickle.load(f)
		
	def train_classifier(self):
		print("Training")

		vehicles_path = '../data/vehicles/**/*.png'
		non_vehicles_path = '../data/non-vehicles/**/*.png'

		vehicles = glob.glob(vehicles_path)
		non_vehicles = glob.glob(non_vehicles_path)

		all_images = vehicles + non_vehicles

		t=time.time()

		
		vehicles_features = self.image_utils.process(all_images)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to extract HOG features...')

		# Create an array stack of feature vectors
		X = np.float64(vehicles_features)#  np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)                        
		# Fit a per-column scaler
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)

		# Define the labels vector
		y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))

		# Split up data into randomized training and test sets
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(
		    scaled_X, y, test_size=0.2, random_state=rand_state)

		print('Feature vector length:', len(X_train[0]))

		
		# Check the training time for the SVC
		t=time.time()
		self.svc.fit(X_train, y_train)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')
		
		# Check the score of the SVC
		print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
		
		with open(self.data_path, "wb") as f:
			pickle.dump(self.svc, f)

	def predict(self, image):
		return self.svc.predict(image)

	def score(self, X, y):
		return self.svc.score(X, y)


def test_full_validation(train=False):
	if train:
		classifier = Classifier(train_data=True)
	else:
		classifier = Classifier()

	vehicles_path = '../data/vehicles/**/*.png'
	non_vehicles_path = '../data/non-vehicles/**/*.png'

	vehicles = glob.glob(vehicles_path)
	non_vehicles = glob.glob(non_vehicles_path)
			
	image_utils = Image_Utils()
	# vehicle_image = image_utils.extract_features_from_image(mpimg.imread(vehicles[10]), hog_channel=0)
	# non_vehicle_image = image_utils.extract_features_from_image(mpimg.imread(non_vehicles[10]), hog_channel=0)
	
	#sample_size = 1000
	#vehicles = vehicles[0:sample_size]
	#non_vehicles = non_vehicles[0:sample_size]

	# non_vehicle_features = []
	# for image in non_vehicles:
	# 	non_vehicle_features.append(image_utils.extract_features_from_image(mpimg.imread(image)))

	# vehicle_features = []
	# for image in vehicles:
	# 	vehicle_features.append(image_utils.extract_features_from_image(mpimg.imread(image)))



	# X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)   

	# vehicles = vehicles[:100]
	# non_vehicles = non_vehicles[:100]

	X = image_utils.process(vehicles + non_vehicles)
	y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))

	accuracy = classifier.score(X, y)
	print('Accuracy:', accuracy)

def try_test_images():
	from matplotlib import pyplot
	classifier = Classifier()
	image_utils = Image_Utils()

	vehicles_path = '../test_images_2/*.png'
	vehicles = glob.glob(vehicles_path)
	vehicle_features = image_utils.process(vehicles)
	# vehicle_features = []
	# for image_path in vehicles:
		# image = mpimg.imread(image_path)
		# pyplot.imshow(image)
		# pyplot.show()
		# vehicle_features.append(image_utils.process(image))

	predictions = classifier.predict(vehicle_features)
	for i, p in enumerate(predictions):
		print(p, vehicles[i])

if __name__ == '__main__':
	test_full_validation()
	try_test_images()
