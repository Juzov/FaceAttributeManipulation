import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Discriminator:
	def __init__(self, seed):
		'''
		This the architechture of the generator
		'''
		self.seed = seed
		self.model = tf.keras.Sequential()

		# Input size (128, 128, 3)
		self.create_layer(
			number_of_maps = 64,
		)

		self.create_layer(
			number_of_maps = 128,
		)

		self.create_layer(
			number_of_maps = 256,
		)

		self.create_layer(
			number_of_maps = 512,
		)

		self.create_layer(
			number_of_maps = 1024,
		)

		# Output size (4, 4, 1)
		self.model.add(
			layers.Conv2D(
				1,
				kernel_size=(4,4),
				strides = 1,
				padding = 'same'
			)
		)

		self.model.add(layers.Flatten())

		self.model.add(
			layers.Dense(
				3,
				activation = 'softmax'
			)
		)

	def __call__(self):
		#do prediction
		pass

	def create_layer(self, number_of_maps, kernel_size=(4,4), stride=2):
		self.model.add(
			layers.Conv2D(
				number_of_maps,
				kernel_size = kernel_size,
				strides = stride,
				padding = 'same',
				kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed)
			)
		)

		self.model.add(layers.LeakyReLU(alpha=0.3))

		self.model.add(layers.BatchNormalization())
