import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from generator import Generator
from discriminator import Discriminator

def process_images(path, filenames):
	dataset = np.zeros((len(filenames), 128, 128, 3), dtype='int64')
	for i, filename in enumerate(filenames):
		im = Image.open(os.path.join(path,filename))
		im = im.resize([128, 128])
		image = np.array(im.convert('RGB'))
		im.close()
		dataset[i] = image
	return dataset

def get_data(size = 1000):
	positive_path = os.path.join('img_align_celeba', 'positives')
	negative_path = os.path.join('img_align_celeba', 'negatives')

	negative_filenames = os.listdir(negative_path)
	positive_filenames = os.listdir(positive_path)

	negative_filenames = random.sample(negative_filenames, size)
	positive_filenames = random.sample(positive_filenames, size)

	negative_labels = [0] * size
	positive_labels = [1] * size

	dataset_negative = process_images(negative_path, negative_filenames)
	dataset_positive = process_images(positive_path, positive_filenames)

	dataset = np.concatenate((dataset_negative, dataset_positive), axis = 0)
	labels = negative_labels + positive_filenames

	indexes = np.array(range(size*2))
	np.random.shuffle(indexes)
	dataset = dataset[indexes]
	labels = labels[indexes]

	return dataset, labels

seed = 9
batch_size = 100
random.seed(seed)
#g_0 = Generator(seed)
#d = Discriminator(seed)

dataset, labels = get_data(1000, batch_size)

print(labels)
