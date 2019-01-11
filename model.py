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

class FaceGAN():
	def __init__(self):
		self.model_name = 'face-gan'
		self.checkpoint_save_dir = 'checkpoints'
		self.checkpoint_load_dir = 'checkpoints'

	def reset_graph(self, seed=42):
		tf.reset_default_graph()
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
		self.Y_neg = tf.placeholder(tf.int32, [batch_size], name="Y_neg")

		self.X_pos = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name="X_pos")
		self.Y_pos = tf.placeholder(tf.int32, [batch_size], name="Y_pos")

		# Convert labels to one_hot, we need this in order to filter out the softmax-prob for the specific label
		Y_neg_oh = tf.one_hot(self.Y_neg, 3)
		Y_pos_oh = tf.one_hot(self.Y_pos, 3)
		Y_fake_oh = tf.one_hot(tf.ones(batch_size, dtype=tf.int32)*2, 3)

		# Get instances to the network-classes
		g_0 = Generator()
		g_1 = Generator()
		d = Discriminator()

		# Residual images/outputs from generators
		r0 = g_0(seed, self.X_neg, "g_0")
		r1 = g_1(seed, self.X_pos, "g_1")

		# The altered image
		x_theta_0 = r0 + self.X_neg
		x_theta_1 = r1 + self.X_pos

		# Discriminator output
		phi_fake_0, _, p_fake_0 = d(seed, x_theta_0)
		phi_fake_1, _, p_fake_1 = d(seed, x_theta_1, True)
		phi_real_0, _, p_real_0 = d(seed, self.X_neg, True)
		phi_real_1, _, p_real_1 = d(seed, self.X_pos, True)

		pt_fake_0 = p_fake_0 * Y_fake_oh
		pt_fake_1 = p_fake_1 * Y_fake_oh
		pt_real_0 = p_real_0 * Y_neg_oh
		pt_real_1 = p_real_1 * Y_pos_oh

		# Dual output
		dual_r_0 = g_1(seed, x_theta_0, "g_1", True)
		dual_r_1 = g_0(seed, x_theta_1, "g_0", True)

		dual_theta_0 = dual_r_0 + x_theta_0
		dual_theta_1 = dual_r_1 + x_theta_1

		_, _, p_dual_0 = d(seed, dual_theta_0, True)
		_, _, p_dual_1 = d(seed, dual_theta_1, True)

		pt_dual_0 = p_dual_0 * Y_neg_oh
		pt_dual_1 = p_dual_1 * Y_pos_oh

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
		# Loss_pix
		loss_pix_0 = tf.reduce_mean(tf.reduce_sum(tf.abs(r0), axis = [1, 2, 3]))
		loss_pix_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(r1), axis = [1, 2, 3]))

		# Loss_cls
		loss_cls_fake_0 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_fake_0, axis = 1) + small_value))
		loss_cls_fake_1 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_fake_1, axis = 1) + small_value))
		loss_cls_real_0 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_real_0, axis = 1) + small_value))
		loss_cls_real_1 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_real_1, axis = 1) + small_value))

		# Loss_per
		loss_per_0 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_real_0 - phi_fake_0), axis=[1, 2, 3]))
		loss_per_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_real_1 - phi_fake_1), axis=[1, 2, 3]))

		# Loss_gan
		loss_gan_0 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_0 * Y_pos_oh, axis = 1) + small_value))
		loss_gan_1 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_fake_1 * Y_neg_oh, axis = 1) + small_value))

		# Loss_dual
		loss_dual_0 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_dual_0, axis = 1) + small_value))
		loss_dual_1 = tf.reduce_mean(-tf.log(tf.reduce_sum(pt_dual_1, axis = 1) + small_value))

		self.all_losses = {
			'loss_pix_0' : loss_pix_0,
			'loss_pix_1' : loss_pix_1,
			'loss_cls_fake_0' : loss_cls_fake_0,
			'loss_cls_fake_1' : loss_cls_fake_1,
			'loss_cls_real_0' : loss_cls_real_0,
			'loss_cls_real_1' : loss_cls_real_1,
			'loss_per_0' : loss_per_0,
			'loss_per_1' : loss_per_1,
			'loss_gan_0' : loss_gan_0,
			'loss_gan_1' : loss_gan_1,
			'loss_dual_0' : loss_dual_0,
			'loss_dual_1' : loss_dual_1,
		}

		""" Final put together losses """
		# Loss for generators
		alpha = 5e-4
		beta = 0.1 * alpha
		self.loss_g_0 = loss_gan_0 + loss_dual_0 + (alpha * loss_pix_0) + (beta * loss_per_0)
		self.loss_g_1 = loss_gan_1 + loss_dual_1 + (alpha * loss_pix_1) + (beta * loss_per_1)

		# Loss for discriminator
		self.loss_d = loss_cls_fake_0 + loss_cls_fake_1 + loss_cls_real_0 + loss_cls_real_1

		# Filter out the variables for each of the networks
		t_vars = tf.trainable_variables()
		g_0_vars = [var for var in t_vars if 'g_0_' in var.name]
		g_1_vars = [var for var in t_vars if 'g_1_' in var.name]
		d_vars = [var for var in t_vars if 'd_' in var.name]

		# Set up optimizers targeting those variables
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			discrimitator_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_discriminator = discrimitator_optimizer.minimize(self.loss_d, var_list = d_vars)

			g_0_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_g_0 = g_0_optimizer.minimize(self.loss_g_0, var_list = g_0_vars)

			g_1_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_g_1 = g_1_optimizer.minimize(self.loss_g_1, var_list = g_1_vars)


	def __call__(self):
		seed = 9
		batch_size = 5
		start_images = 0
		number_of_images = 1400
		train_ratio= 0.7

		self.reset_graph(seed)

		self.build_model(seed, batch_size)

		init = tf.global_variables_initializer()

		writer = tf.summary.FileWriter("graphs", tf.get_default_graph())

		self.saver = tf.train.Saver(max_to_keep=2)

		with tf.Session() as self.sess:
			# restore check-point if it exits
			if (True):
				could_load, checkpoint_counter = self.load()
			else:
				could_load = False

			if could_load:
				#
				start_images = 0
				print(" [*] Load SUCCESS")
			else:
				self.sess.run(init)
				start_images = 0
				print(" [!] Load failed...")

			if ((start_images + number_of_images) > 9817):
				print('not enough images')
				sys.exit()

			data_neg, data_pos = get_data(start_images, number_of_images, train_ratio, seed)

			n_epochs = 35

			for epoch in range(n_epochs):
				# TODO - this fails
				print(f'epoch{epoch}')
				for i in range(int(np.floor(float(number_of_images*train_ratio)/batch_size))):
					batch_pos = data_pos['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
					batch_neg = data_neg['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
					labels_pos = data_pos['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]
					labels_neg = data_neg['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]

					# train discriminator
					loss, _ = self.sess.run(
						[self.loss_d, self.train_step_discriminator],
						feed_dict = {
							self.X_neg : batch_neg,
							self.X_pos : batch_pos,
							self.Y_neg : labels_neg,
							self.Y_pos : labels_pos
						}
					)
					print(f'Loss for discriminator is: {loss}')

					# train generator_0
					loss, _ = self.sess.run(
						[self.loss_g_0, self.train_step_g_0],
						feed_dict = {
							self.X_neg : batch_neg,
							self.X_pos : batch_pos,
							self.Y_neg : labels_neg,
							self.Y_pos : labels_pos
						}
					)
					print(f'Loss for g_0 is: {loss}')

					# train generator_1
					loss, _ = self.sess.run(
						[self.loss_g_1, self.train_step_g_1],
						feed_dict = {
							self.X_neg : batch_neg,
							self.X_pos : batch_pos,
							self.Y_neg : labels_neg,
							self.Y_pos : labels_pos
						}
					)
					print(f'Loss for g_1 is: {loss}')

					all_loss, all_images = self.sess.run(
						[self.all_losses, self.all_images],
						feed_dict = {
							self.X_neg : batch_neg,
							self.X_pos : batch_pos,
							self.Y_neg : labels_neg,
							self.Y_pos : labels_pos
						}
					)
					print(f'all losses: {all_loss}')

					for key, img_collection in all_images.items():
						for j,img in enumerate(img_collection):
							im = Image.fromarray(scale_image(img).astype('uint8'))
							im.save(f'images/{key}-{str(j)}.png')
							im.close()

				if (epoch % 2 == 0):
					self.save(start_images + ((i*batch_size)+batch_size))

fg = FaceGAN()
fg()
