import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
from time import time
from tensorflow.keras import layers
from generator import Generator
from discriminator import Discriminator
from utils import get_data, scale_image
from tensorflow.keras.callbacks import TensorBoard as tb
from tensorflow.python.tools import inspect_checkpoint as chkp
import scipy.misc as misc

class FaceGAN():
	def __init__(self):
		self.model_name = 'face-gan'
		self.checkpoint_save_dir = 'checkpoints'
		self.checkpoint_load_dir = 'checkpoints'

	def reset_graph(self, seed=42):
		tf.set_random_seed(seed)
		np.random.seed(seed)

	def save(self, step):

		if not os.path.exists(self.checkpoint_save_dir):
			os.makedirs(self.checkpoint_save_dir)

		self.saver.save(self.sess,os.path.join(self.checkpoint_save_dir, self.model_name+'.model'), global_step=step)

	def load(self):
		import re
		print(" [*] Reading checkpoints...")

		ckpt = tf.train.get_checkpoint_state(self.checkpoint_load_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			#chkp.print_tensors_in_checkpoint_file(os.path.join(self.checkpoint_dir, ckpt_name), tensor_name='', all_tensors=True)
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def build_model(self, seed, batch_size):
		self.X_neg = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name="X_neg")
		self.X_pos = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name="X_pos")

		one_hot_fake = np.ones([self.batchsize, 1])
		one_hot_fake = np.float32(one_hot_fake == np.array([1, 2, 3]))
		one_hot_pos = np.ones([self.batchsize, 1]) * 2
		one_hot_pos = np.float32(one_hot_pos == np.array([1, 2, 3]))
		one_hot_neg = np.ones([self.batchsize, 1]) * 3
		one_hot_neg = np.float32(one_hot_neg == np.array([1, 2, 3]))

		# Get instances to the network-classes
		g_0 = Generator(seed, 'g_0')
		g_1 = Generator(seed, 'g_1')
		d = Discriminator(seed)

		# Residual images/outputs from generators
		r0 = g_0(self.X_neg)
		r1 = g_1(self.X_pos)

		# The altered image
		x_theta_0 = r0 + self.X_neg
		x_theta_1 = r1 + self.X_pos

		# Discriminator output
		phi_fake_0, p_fake_0 = d(x_theta_0)
		phi_fake_1, p_fake_1 = d(x_theta_1, True)
		phi_real_0, p_real_0 = d(self.X_neg, True)
		phi_real_1, p_real_1 = d(self.X_pos, True)

		# Dual output
		phi_dual_x0_theta, dual_x0_theta = d(g_1(x_theta_0, True) + x_theta_0, True)
		phi_dual_x1_theta, dual_x1_theta = d(g_0(x_theta_1, True) + x_theta_1, True)

		self.all_images = {
			'neg_org' : self.X_neg,
			'pos_org' : self.X_pos,
			'res_add' : r0,
			'res_rem' : r1,
			'added'   : x_theta_0,
			'removed' : x_theta_1,
		}

		""" LOSSES """
		small_value = 1e-9 # we need this to avoid nan/inf
		# Loss pix
		self.l_pix0 = tf.reduce_mean(tf.reduce_sum(tf.abs(r0), axis=[1, 2, 3]))
		self.l_pix1 = tf.reduce_mean(tf.reduce_sum(tf.abs(r1), axis=[1, 2, 3]))

		# Loss_cls
		#Discriminator loss
		self.l_cls = tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_0 * one_hot_fake, axis=1) + small_value))
		self.l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_1 * one_hot_fake, axis=1) + small_value))
		self.l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_real_1 * one_hot_pos, axis=1) + small_value))
		self.l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_real_0 * one_hot_neg, axis=1) + small_value))

		# perceptual loss
		self.l_per0 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_real_0 - phi_fake_0), axis=[1, 2, 3]))
		self.l_per1 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_real_1 - phi_fake_1), axis=[1, 2, 3]))

		# gan loss
		self.l_gan0 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_0 * one_hot_pos, axis=1) + small_value))
		self.l_gan1 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_1 * one_hot_fake, axis=1) + small_value))

		#dual loss
		self.l_dual0 = tf.reduce_mean(-tf.log(tf.reduce_sum(dual_x0_theta * one_hot_fake, axis=1)+ small_value))
		self.l_dual1 = tf.reduce_mean(-tf.log(tf.reduce_sum(dual_x1_theta * one_hot_pos, axis=1) + small_value))

		alpha = 5e-5
		beta = 0.1 * alpha
		self.l_g0 = self.l_gan0 + self.l_dual0 + alpha * self.l_pix0 + beta * self.l_per0
		self.l_g1 = self.l_gan1 + self.l_dual1 + alpha * self.l_pix1 + beta * self.l_per1

		self.train_g0 = tf.train.AdamOptimizer(2e-4).minimize(self.l_g0, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "g_0"))
		self.train_g1 = tf.train.AdamOptimizer(2e-4).minimize(self.l_g1, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "g_1"))
		self.train_d = tf.train.AdamOptimizer(2e-4).minimize(self.l_cls, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "d_"))

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def __call__(self):
		seed = 9
		self.batchsize = batch_size = 5
		start_images = 0
		number_of_images = 1400
		train_ratio= 0.7

		self.reset_graph(seed)

		self.build_model(seed, batch_size)

		# Set up a saver to load and save the model and a tensorboard graph
		self.saver = tf.train.Saver(max_to_keep=2)
		writer = tf.summary.FileWriter("graphs", tf.get_default_graph())

		# Check if we should load a model, and if we succeded doing so
		if (True):
			could_load, checkpoint_counter = self.load()
		else:
			could_load = False

		if could_load:
			#
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		# Load the data (both train and test)
		data_neg, data_pos = get_data(start_images, number_of_images, train_ratio, seed)

		n_epochs = 35

		for epoch in range(n_epochs):
			# TODO - this fails
			print(f'epoch{epoch}')
			for i in range(int(np.floor(float(number_of_images*train_ratio)/batch_size))):
				# Get the current batch
				batch_pos = data_pos['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
				batch_neg = data_neg['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
				labels_pos = data_pos['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]
				labels_neg = data_neg['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]

				# Update the weights
				self.sess.run(
					self.train_d,
					feed_dict = {
						self.X_neg: batch_neg,
						self.X_pos: batch_pos
					}
				)
				self.sess.run(
					self.train_g0,
					feed_dict = {
						self.X_neg: batch_neg,
						self.X_pos: batch_pos
					}
				)
				self.sess.run(
					self.train_g1,
					feed_dict = {
						self.X_neg: batch_neg,
						self.X_pos: batch_pos
					}
				)

				if i % 10 == 0:
					[l_pix0, l_pix1, l_per0, l_per1, l_gan0, l_gan1, l_dual0, l_dual1, l_D, l_g0, l_g1] = \
					self.sess.run(
						[self.l_pix0, self.l_pix1, self.l_per0, self.l_per1, self.l_gan0, self.l_gan1, self.l_dual0, self.l_dual1, self.l_cls, self.l_g0, self.l_g1],
						feed_dict = {
							self.X_neg: batch_neg,
							self.X_pos: batch_pos
						}
					)
					print("epoch: %d batch: %d l_pix0: %g l_pix1: %g l_per0: %g l_per1: %g l_gan0: %g l_gan1: %g l_dual0: %g l_dual1: %g l_g0: %g l_g1: %g l_D: %g"
						% (epoch, i, l_pix0, l_pix1, l_per0, l_per1, l_gan0, l_gan1, l_dual0, l_dual1, l_g0, l_g1, l_D))

			if (epoch % 10 == 0):
				self.save(epoch)
				all_images = self.sess.run(
					self.all_images,
					feed_dict = {
						self.X_neg : batch_neg,
						self.X_pos : batch_pos,
					}
				)

				for key, img_collection in all_images.items():
					for j,img in enumerate(img_collection):
						im = Image.fromarray(scale_image(img).astype('uint8'))
						im.save(f'images/{key}-{str(j)}.png')
						im.close()

fg = FaceGAN()
fg()
