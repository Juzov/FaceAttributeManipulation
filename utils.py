import os
from PIL import Image
import random
import numpy as np

def process_images(filenames, attribute):
	dataset = np.zeros((len(filenames), 128, 128, 3), dtype='int64')
	for i, filename in enumerate(filenames):
		path = ''
		if attribute == 0:
			path = os.path.join('img_align_celeba', 'negatives')
		else:
			path = os.path.join('img_align_celeba', 'positives')

		im = Image.open(os.path.join(path,filename))
		im = im.resize([128, 128])
		image = np.array(im.convert('RGB'))
		im.close()
		dataset[i] = image
	return dataset

def get_attribute_training_test(size, training_ratio, seed, filenames, attribute):
	labels = [attribute] * size
	training_size = int(np.floor(0.7*size))
	filenames_train = filenames[:training_size]
	filenames_test = filenames[training_size:]
	labels_train = np.array(labels[:training_size])
	labels_test = np.array(labels[training_size:])


	dataset_train = process_images(filenames_train, attribute)
	dataset_test = process_images(filenames_test, attribute)

	X = {
		'train_data'    :   dataset_train,
		'train_labels'  :   labels_train,
		'test_data'     :   dataset_test,
		'test_labels'   :   labels_test,
	}

	return X


def get_data(size = 1000, training_ratio=0.7, seed=9):
	random.seed(seed)

	positive_path = os.path.join('img_align_celeba', 'positives')
	negative_path = os.path.join('img_align_celeba', 'negatives')

	negative_filenames = os.listdir(negative_path)
	positive_filenames = os.listdir(positive_path)

	negative_filenames = random.sample(negative_filenames, size)
	positive_filenames = random.sample(positive_filenames, size)

	X_neg = get_attribute_training_test(size, training_ratio, seed, negative_filenames, 0)
	X_pos = get_attribute_training_test(size, training_ratio, seed, positive_filenames, 1)

	return X_neg, X_pos
