import numpy as np
import tensorflow as tf

class Generator:

	def __call__(self, seed, data, gen_name, reuse = False):
		'''
		This the architechture of the generator
		'''

		with tf.variable_scope(gen_name, reuse=reuse):

			self.seed = seed

			# Input size (128, 128, 3)
			conv1 = self.conv2d(
				variable_scope = gen_name,
				input = data,
				filters = 64,
				kernel_size = [5,5],
				stride = 1,
				name = gen_name + '_conv_1'
			)

			bn1 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv1,
					epsilon=1e-5,
					scale=True,
					scope= gen_name + '_bn_1'
				)
			)

			conv2 = self.conv2d(
				variable_scope = gen_name,
				input = bn1,
				filters = 128,
				kernel_size = [4,4],
				stride = 2,
				name = gen_name + '_conv_2'
			)

			bn2 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv2,
					epsilon=1e-5,
					scale=True,
					scope= gen_name + '_bn_2'
				)
			)

			conv3 = self.conv2d(
				variable_scope = gen_name,
				input = bn2,
				filters = 256,
				kernel_size = [4,4],
				stride = 2,
				name = gen_name + '_conv_3'
			)

			bn3 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					conv3,
					epsilon=1e-5,
					scale=True,
					scope= gen_name + '_bn_3'
				)
			)

			deconv4 = self.deconv2d(
				variable_scope = gen_name,
				input = bn3,
				filters = 128,
				kernel_size = [3,3],
				stride = 2,
				name = gen_name + '_deconv_4'
			)

			bn4 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					deconv4,
					epsilon=1e-5,
					scale=True,
					scope= gen_name + '_bn_4'
				)
			)

			deconv5 = self.deconv2d(
				variable_scope = gen_name,
				input = bn4,
				filters = 64,
				kernel_size = [3,3],
				stride = 2,
				name = gen_name + '_deconv_5'
			)

			bn5 = tf.nn.leaky_relu(
				tf.contrib.layers.instance_norm(
					deconv5,
					epsilon=1e-5,
					scale=True,
					scope= gen_name + '_bn_5'
				)
			)

			# Output size (128, 128, 3)
			self.output = self.conv2d(
				variable_scope = gen_name,
				input = bn5,
				filters = 3,
				kernel_size=[4,4],
				stride = 1,
				name = gen_name + '_conv_6'
			)

			return self.output

	def conv2d(self, variable_scope, input, filters, kernel_size, stride, name):

		w = tf.get_variable(name + 'w', [kernel_size[0], kernel_size[1], input.get_shape()[-1], filters],
			initializer=tf.truncated_normal_initializer(stddev = 0.02, seed = self.seed))

		conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name=name)

		biases = tf.get_variable(name + 'biases', [filters], initializer=tf.constant_initializer(0.0))

		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

	def deconv2d(self, variable_scope, input, filters, kernel_size, stride, name):
		w = tf.get_variable(name + 'w', [kernel_size[0], kernel_size[1], filters, input.get_shape()[-1]],
			initializer=tf.truncated_normal_initializer(stddev = 0.02, seed = self.seed))

		conv = tf.nn.conv2d_transpose(input, w, output_shape=tf.convert_to_tensor([input.get_shape()[0].value, input.get_shape()[1].value*stride, input.get_shape()[2].value*stride, filters]), strides=[1, stride, stride, 1], padding='SAME', name=name)

		biases = tf.get_variable(name + 'biases', [filters], initializer=tf.constant_initializer(0.0))

		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

