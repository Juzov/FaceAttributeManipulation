import numpy as np
import tensorflow as tf

class Discriminator:
	def __call__(self, seed, data, reuse = False):
		'''
		This the architechture of the generator
		'''
		with tf.variable_scope("d_", reuse=reuse):
			self.seed = seed

			# Input size (128, 128, 3)
			conv1 = self.conv2d(
				variable_scope = 'd_',
				input = data,
				filters = 64,
				name = 'd_conv1'
			)

			bn1 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv1,
					epsilon=1e-5,
					scale=True,
					scope= 'd_bn_1'
				)
			)

			conv2 = self.conv2d(
				variable_scope = 'd_',
				input = bn1,
				filters = 128,
				name = 'd_conv2'
			)

			bn2 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv2,
					epsilon=1e-5,
					scale=True,
					scope= 'd_bn_2'
				)
			)

			self.phi = self.conv2d(
				variable_scope = 'd_',
				input = bn2,
				filters = 256,
				name = 'd_conv3'
			)

			bn3 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					self.phi,
					epsilon=1e-5,
					scale=True,
					scope= 'd_bn_3'
				)
			)

			conv4 = self.conv2d(
				variable_scope = 'd_',
				input = bn3,
				filters = 512,
				name = 'd_conv4'
			)

			bn4 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv4,
					epsilon=1e-5,
					scale=True,
					scope= 'd_bn_4'
				)
			)

			conv5 = self.conv2d(
				variable_scope = 'd_',
				input = conv4,
				filters = 1024,
				name = 'd_conv5'
			)

			bn5 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv5,
					epsilon=1e-5,
					scale=True,
					scope= 'd_bn_5'
				)
			)

			self.logits = conv6 = self.conv2d(
				variable_scope = 'd_',
				input = bn5,
				filters = 1,
				name = 'd_conv6',
				stride = 1
			)

			batch_size = data.get_shape()[0].value
			flatten7 = tf.reshape(conv6, [batch_size, -1])

			dense_w = tf.get_variable("d_dense_8_w", [flatten7.get_shape()[1].value, 3], tf.float32,
				tf.truncated_normal_initializer(stddev = 0.02, seed = self.seed))

			dense_bias = tf.get_variable("d_dense_8_bias", [3],
				initializer=tf.constant_initializer(0))


			self.output = tf.nn.softmax(tf.matmul(flatten7, dense_w) + dense_bias)

			return self.phi, self.logits, self.output

	def conv2d(self, variable_scope, input, filters, name, kernel_size = [4,4], stride = 2):
		w = tf.get_variable(name + 'w', [kernel_size[0], kernel_size[1], input.get_shape()[-1], filters],
			initializer=tf.truncated_normal_initializer(stddev = 0.02, seed = self.seed))

		conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name=name)

		biases = tf.get_variable(name + 'biases', [filters], initializer=tf.constant_initializer(0.0))

		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv


