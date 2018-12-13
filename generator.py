import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Generator:
	def __init__(self, seed):
		'''
		This the architechture of the generator
		'''
		self.seed = seed
		self.model = tf.keras.Sequential()

		# Input size (128, 128, 3)
		self.create_layer(
			function = layers.Conv2D,
			number_of_maps = 64,
			kernel_size = (5,5),
			stride = 1
		)

		self.create_layer(
			function = layers.Conv2D,
			number_of_maps = 128,
			kernel_size = (4,4),
			stride = 2
		)

		self.create_layer(
			function = layers.Conv2D,
			number_of_maps = 256,
			kernel_size = (4,4),
			stride = 2
		)

		self.create_layer(
			function = layers.Conv2DTranspose,
			number_of_maps = 128,
			kernel_size = (3,3),
			stride = 1
		)

		self.create_layer(
			function = layers.Conv2DTranspose,
			number_of_maps = 64,
			kernel_size = (3,3),
			stride = 1
		)

		# Output size (128, 128, 3)
		self.model.add(
			layers.Conv2D(
				3,
				kernel_size=(4,4),
				strides = 1,
				padding = 'same'
			)
		)

	def __call__(self):
		#do prediction
		output = self.model.predict_on_batch(x)
		return output

	def create_layer(self, function, number_of_maps, kernel_size, stride):
		self.model.add(
			function(
				number_of_maps,
				kernel_size=kernel_size,
				strides = stride,
				padding = 'same',
				kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed)
			)
		)

		self.model.add(layers.LeakyReLU(alpha=0.3))

		self.model.add(layers.BatchNormalization())


