import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt


CROWDAI_DATASET_PATH = '../data/crowdai/labels.csv'
AUTTI_DATASET_PATH = '../data/autti/labels.csv'

def readCrowdAIData():
	# Format: xmin,xmax,ymin,ymax,Frame,Label,Preview URL
	with open(CROWDAI_DATASET_PATH, 'r') as f:
		reader = csv.reader(f)
		crowdai_data = list(reader)

	default_image = cv2.imread("../data/crowdai/{}".format(crowdai_data[0][4]))
	heat = np.zeros_like(default_image[:,:,0]).astype(np.float)
	xs = []
	ys = []
	count = 0
	print("First Image Shape", default_image.shape)
	for (xmin,xmax,ymin,ymax,frame,label,preview) in crowdai_data:
		xmin = int(xmin)
		xmax = int(xmax)
		ymin = int(ymin)
		ymax = int(ymax)
		if label == 'Car' or label == 'Truck':
			# image = cv2.imread("../data/crowdai/{}".format(frame))
			# if default_image.shape != image.shape:
			# 	print(frame, image.shape)
			count += 1
			heat[int(xmax):int(ymax),int(xmin):int(ymin)] += 1
			xs.append(abs(xmax-xmin))
			ys.append(abs(ymax-ymin))

	print(count, "files processed")
	heat[heat <= 1] = 0
	heatmap = np.clip(heat, 0, 255)
	plt.imshow(heatmap, cmap='hot')
	plt.show()
	# plt.title("Image Size: {}, {}".format(image.shape))
	plt.scatter(xs, ys)
	plt.show()

def readAuttiData():
	# Format: Frame,xmin,ymin,xmax,ymax,something,label
	with open(AUTTI_DATASET_PATH, 'r') as f:
		reader = csv.reader(f, delimiter=' ')
		autti_data = list(reader)

	image = cv2.imread("../data/autti/{}".format(autti_data[0][0]))
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	xs = []
	ys = []
	count = 0
	#(frame,xmin,ymin,xmax,ymax,blah,label, etc)
	for data in autti_data:
		xmin = int(data[1])
		xmax = int(data[2])
		ymin = int(data[3])
		ymax = int(data[4])
		if data[6] == 'car' or data[6] == 'truck':
			count += 1
			heat[int(xmax):int(ymax),int(xmin):int(ymin)] += 1
			xs.append(abs(xmax-xmin))
			ys.append(abs(ymax-ymin))

	print(count, "files processed")
	heat[heat <= 1] = 0
	heatmap = np.clip(heat, 0, 255)
	plt.imshow(heatmap, cmap='hot')
	plt.show()
	plt.scatter(xs, ys)
	plt.show()


# readAuttiData()
readCrowdAIData()