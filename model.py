import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from generator import Generator
from discriminator import Discriminator
from utils import get_data


seed = 9
batch_size = 100

g_0 = Generator(seed)
g_1 = Generator(seed)
d = Discriminator(seed)

dataset_train, dataset_test, labels_train, labels_test = get_data(1000, 0.7, seed)

epochs = 1
for epoch in range(epochs):
	for i in range(int(np.floor(len(dataset_train)/batch_size))):
		batch = dataset_train[i*batch_size:(i*batch_size)+batch_size]
		batch_labels = labels_train[i*batch_size:(i*batch_size)+batch_size]

		r_0 = g_0(batch)
		r_1 = g_1(batch)
		x_tilde_0 = np.add(r_0, batch)
		x_tilde_1 = np.add(r_1, batch)

		discrimitator_batch = np.concatenate((batch, x_tilde_0, x_tilde_1), axis=0)
		discrimitator_labels = np.empty(len(x_tilde_0) + len(x_tilde_1))
		discrimitator_labels = np.concatenate((batch_labels, discrimitator_labels), axis=0)

		discrimitator_loss = d.model.train_on_batch(discrimitator_batch, discrimitator_labels)