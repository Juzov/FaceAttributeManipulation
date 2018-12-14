import numpy as np
import tensorflow as tf

class Generator:
	def __init__(self, seed, data):
		'''
		This the architechture of the generator
		'''
		self.seed = seed

		# Input size (128, 128, 3)
		conv1 = self.create_layer(
			inputs = data,
			function = tf.layers.conv2d,
			filters = 64,
			kernel_size = [5,5],
			stride = 1
		)

		conv2 = self.create_layer(
			inputs = conv1,
			function = tf.layers.conv2d,
			filters = 128,
			kernel_size = [4,4],
			stride = 2
		)

		conv3 = self.create_layer(
			inputs = conv2,
			function = tf.layers.conv2d,
			filters = 256,
			kernel_size = [4,4],
			stride = 2
		)

		deconv4 = self.create_layer(
			inputs = conv3,
			function = tf.layers.conv2d_transpose,
			filters = 128,
			kernel_size = [3,3],
			stride = 2
		)

		deconv5 = self.create_layer(
			inputs = deconv4,
			function = tf.layers.conv2d_transpose,
			filters = 64,
			kernel_size = [3,3],
			stride = 2
		)

		# Output size (128, 128, 3)
		self.output = tf.layers.conv2d(
			inputs = deconv5,
			filters = 3,
			kernel_size=[4,4],
			strides = 1,
			padding = 'same',
			kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed)
		)

	def create_layer(self, inputs, function, filters, kernel_size, stride):
		first_layer = function(
			inputs = inputs,
			filters = filters,
			kernel_size=kernel_size,
			strides = stride,
			padding = 'same',
			kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed),
			activation = tf.nn.leaky_relu
		)

		second_layer = tf.layers.batch_normalization(inputs = first_layer)

		return second_layer


