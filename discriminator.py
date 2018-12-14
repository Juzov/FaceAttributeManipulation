import numpy as np
import tensorflow as tf

class Discriminator:
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
		)

		conv2 = self.create_layer(
			inputs = conv1,
			function = tf.layers.conv2d,
			filters = 128
		)

		self.phi = self.create_layer(
			inputs = conv2,
			function = tf.layers.conv2d,
			filters = 256
		)

		conv4 = self.create_layer(
			inputs = self.phi,
			function = tf.layers.conv2d,
			filters = 512
		)

		conv5 = self.create_layer(
			inputs = conv4,
			function = tf.layers.conv2d,
			filters = 1024
		)

		conv6 = tf.layers.conv2d(
			inputs = conv5,
			filters = 1,
			kernel_size=[4,4],
			strides = 1,
			padding = 'same',
			kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed)
		)

		flatten7 = tf.layers.flatten(inputs = conv6)

		self.output = tf.layers.dense(
			inputs = flatten7,
			units = 3,
			activation = tf.nn.softmax,
			kernel_initializer = tf.keras.initializers.he_normal(seed=self.seed)
		)

	def create_layer(self, inputs, function, filters, kernel_size = [4,4], stride = 2):
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

