import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from generator import Generator
from discriminator import Discriminator
from utils import get_data


seed = 9
batch_size = 100
number_of_images = 1000

g_0 = Generator(seed)
g_1 = Generator(seed)
d = Discriminator(seed)

X_neg, X_pos = get_data(number_of_images, 0.7, seed)

epochs = 1
for epoch in range(epochs):
	for i in range(int(np.floor(number_of_images/batch_size))):
		batch_pos = X_pos['train_data'][i*batch_size:(i*batch_size)+batch_size]
		batch_neg = X_neg['train_data'][i*batch_size:(i*batch_size)+batch_size]
		labels_pos = X_pos['train_labels'][i*batch_size:(i*batch_size)+batch_size]
		labels_neg = X_neg['train_labels'][i*batch_size:(i*batch_size)+batch_size]
		batch = np.concatenate((batch_pos, batch_neg), axis = 0)
		batch_labels = np.concatenate((labels_pos, labels_neg), axis = 0)

		r_0 = g_0(batch_pos)
		r_1 = g_1(batch_neg)
		x_tilde_0 = np.add(r_0, batch_pos)
		x_tilde_1 = np.add(r_1, batch_neg)


		discrimitator_batch = np.concatenate((batch, x_tilde_0, x_tilde_1), axis=0)
		discrimitator_labels = np.empty(len(x_tilde_0) + len(x_tilde_1))
		discrimitator_labels = np.concatenate((batch_labels, discrimitator_labels), axis=0)

		discrimitator_loss = d.model.train_on_batch(discrimitator_batch, discrimitator_labels)

		print(f'Loss on discriminator{discrimitator_loss}')