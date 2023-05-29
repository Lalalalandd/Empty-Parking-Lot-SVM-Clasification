from turtle import window_height
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense


from imutils import paths
import numpy as np
import pickle
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/", help='Root directory of data')
args = parser.parse_args()

# load the network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None

# loop over the data splits
for split in ("clf-data", "DUMMY-clf"):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([args.root, split])
	imagePaths = list(paths.list_images(p))

	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	#random.shuffle(imagePaths)
	labels = [p.split(os.path.sep)[-2] for p in imagePaths]

	# if the label encoder is None, create it
	if le is None:
		le = LabelEncoder()
		le.fit(labels)

	# open the output CSV file for writing
	csvPath = os.path.sep.join(["VGG16_output",
		"{}.csv".format(split)])
	csv = open(csvPath, "w")

	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(imagePaths), 32)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(imagePaths) / float(32)))))
		batchPaths = imagePaths[i:i + 32]
		batchLabels = le.transform(labels[i:i + 32])
		batchImages = []

		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility
			# while ensuring the image is resized to 224x224 pixels
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)

			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=32)
		features = features.reshape((features.shape[0], -1))
        
        

		# loop over the class labels and extracted features
		for (label, vec) in zip(batchLabels, features):
			# construct a row that exists of the class label and
			# extracted features
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(vec, label))

	# close the CSV file
	csv.close()
           
# serialize the label encoder to disk
f = open("VGG16_output/le.cpickle", "wb")
f.write(pickle.dumps(le))
f.close()
#print(features.size)
