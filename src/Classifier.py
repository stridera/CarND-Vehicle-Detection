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
	def __init__(self, train_data=False):
		self.data_path='../data/classifier.pickle'	

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

		t=time.time()

		image_utils = Image_Utils()
		vehicles_features = image_utils.extract_features(vehicles)
		non_vehicles_features = image_utils.extract_features(non_vehicles)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to extract HOG features...')

		# Create an array stack of feature vectors
		X = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)                        
		# Fit a per-column scaler
		print(X.shape)
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)

		# Define the labels vector
		y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(non_vehicles_features))))

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
		
		# Check the prediction time for a single sample
		t=time.time()
		n_predict = 10
		print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
		print('For these',n_predict, 'labels: ', y_test[0:n_predict])
		t2 = time.time()
		print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

		with open(self.data_path, "wb") as f:
			pickle.dump(self.svc, f)

	def predict(self, image):
		return self.svc.predict(image)

if __name__ == '__main__':
	train = False
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
	
	sample_size = 1000
	vehicles = vehicles[0:sample_size]
	non_vehicles = non_vehicles[0:sample_size]

	non_vehicle_features = []
	for image in non_vehicles:
		non_vehicle_features.append(image_utils.extract_features_from_image(mpimg.imread(image)))

	prediction = classifier.predict(non_vehicle_features)
	incorrect = np.count_nonzero(prediction)
	print(incorrect, "of", len(non_vehicle_features), 'failed')


	vehicle_features = []
	for image in vehicles:
		vehicle_features.append(image_utils.extract_features_from_image(mpimg.imread(image)))

	prediction = classifier.predict(vehicle_features)
	correct = np.count_nonzero(prediction)
	print(correct, "of", len(vehicle_features), 'passed')

