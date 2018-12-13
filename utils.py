import os
from PIL import Image
import random
import numpy as np

def process_images(filenames, labels):
	dataset = np.zeros((len(filenames), 128, 128, 3), dtype='int64')
	for i, filename in enumerate(filenames):
		path = ''
		if labels[i] == 0:
			path = os.path.join('img_align_celeba', 'negatives')
		else:
			path = os.path.join('img_align_celeba', 'positives')

		im = Image.open(os.path.join(path,filename))
		im = im.resize([128, 128])
		image = np.array(im.convert('RGB'))
		im.close()
		dataset[i] = image
	return dataset

def get_data(size = 1000, training_ratio=0.7, seed=9):
	random.seed(seed)

	positive_path = os.path.join('img_align_celeba', 'positives')
	negative_path = os.path.join('img_align_celeba', 'negatives')

	negative_filenames = os.listdir(negative_path)
	positive_filenames = os.listdir(positive_path)

	negative_filenames = random.sample(negative_filenames, size)
	positive_filenames = random.sample(positive_filenames, size)

	negative_labels = [0] * size
	positive_labels = [1] * size
	filenames = np.array(negative_filenames + positive_filenames)
	labels = np.array(negative_labels + positive_labels)

	indexes = np.array(range(size*2), dtype='int64')
	np.random.shuffle(indexes)
	training_size = int(np.floor(0.7*size*2))
	indexes_train = indexes[:training_size]
	indexes_test = indexes[training_size:]
	filenames_train = filenames[indexes_train]
	filenames_test = filenames[indexes_test]
	labels_train = labels[indexes_train]
	labels_test = labels[indexes_test]


	dataset_train = process_images(filenames_train, labels_train)
	dataset_test = process_images(filenames_test, labels_test)

	return dataset_train, dataset_test, labels_train, labels_test