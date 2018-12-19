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
		self.checkpoint_load_dir = '/floyd/input/checkpoints'

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
		self.X_neg = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
		self.Y_neg = tf.placeholder(tf.int32, [batch_size])

		self.X_pos = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
		self.Y_pos = tf.placeholder(tf.int32, [batch_size])

		g_0 = Generator()
		g_1 = Generator()
		d = Discriminator()

		r_0 = g_0(seed, self.X_neg, "g_0", False)
		r_1 = g_1(seed, self.X_pos, "g_1", False)

		x_tilde_0 = tf.add(r_0, self.X_neg)
		self.test_image_1 = r_0[0]
		self.test_image = x_tilde_0[0]
		self.test_image_2 = self.X_neg[0]
		x_tilde_1 = tf.add(r_1, self.X_pos)
		self.test_image_3 = r_1[0]
		self.test_image_4 = x_tilde_1[0]

		discriminator_fake_input = tf.concat([x_tilde_0,x_tilde_1], axis = 0)
		discriminator_real_input = tf.concat([self.X_neg, self.X_pos], axis = 0)

		_, y_hat_fake_logits, y_hat_fake = d(seed, discriminator_fake_input, False)
		_, y_hat_real_logits, y_hat_real = d(seed, discriminator_real_input, True)

		"""
			DISCRIMINATOR LOSS
		"""
		# Cls loss
		fake_labels = tf.fill([discriminator_fake_input.shape[0].value], 2)
		real_labels = tf.concat([self.Y_neg, self.Y_pos], axis = 0)
		# real_one_hot = tf.one_hot(real_labels, depth = 3)


		cls_labels = tf.concat([real_labels, fake_labels], axis = 0)
		# cls_labels = tf.one_hot(cls_labels, depth = 3)

		cls_data = tf.concat([y_hat_real_logits, y_hat_fake_logits], axis = 0)

		self.loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = cls_labels,
			logits = cls_data
		))

		"""
			GENERATOR LOSS
		"""

		# Perception loss
		phi_fake_0, _, _ = d(seed, x_tilde_0, True)
		phi_fake_1, _, _ = d(seed, x_tilde_1, True)
		phi_real_0, _, _ = d(seed, self.X_neg, True)
		phi_real_1, _, _ = d(seed, self.X_pos, True)
		self.loss_per_0 = tf.losses.absolute_difference(
			labels = phi_real_0,
			predictions = phi_fake_0
		)
		self.loss_per_1 = tf.losses.absolute_difference(
			labels = phi_real_1,
			predictions = phi_fake_1
		)
		# Pix loss
		self.loss_pix_0 = tf.reduce_mean(tf.reduce_sum(tf.abs(r_0), axis=[1, 2, 3]))
		self.loss_pix_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(r_1), axis=[1, 2, 3]))

		# Dual loss
		r_0_reverse = g_0(seed, x_tilde_1, "g_0", True)
		r_1_reverse = g_1(seed, x_tilde_0, "g_1", True)

		x_tilde_0_dual = tf.add(r_0_reverse, x_tilde_1)
		x_tilde_1_dual = tf.add(r_1_reverse, x_tilde_0)

			#input for g0 loss
		_, logits_dual_0, _ = d(seed, x_tilde_1_dual, True)
			#input for g1 loss
		_, logits_dual_1, _ = d(seed, x_tilde_0_dual, True)

		logits_dual = tf.concat([logits_dual_0, logits_dual_1], axis=0)

			#loss
		self.loss_dual_0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = self.Y_neg,
			logits = logits_dual_0
		))
		self.loss_dual_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = self.Y_pos,
			logits = logits_dual_1
		))

		# GAN loss
		_, logits_gan_0, _ = d(seed, x_tilde_0, True)
		_, logits_gan_1, _ = d(seed, x_tilde_1, True)

		#logits_gan = tf.concat([logits_gan_0, logits_gan_1], axis=0)

			#loss
		self.loss_gan_0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = self.Y_neg,
			logits = logits_gan_0
		))
		self.loss_gan_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = self.Y_pos,
			logits = logits_gan_1
		))

		# Generator loss
		self.loss_g_0 = self.loss_gan_0 + self.loss_dual_0 + (5e-4 * self.loss_pix_0) + (5e-5 * self.loss_per_0)
		self.loss_g_1 = self.loss_gan_1 + self.loss_dual_1 + (5e-4 * self.loss_pix_1) + (5e-5 * self.loss_per_1)

		t_vars = tf.trainable_variables()
		g_0_vars = [var for var in t_vars if 'g_0_' in var.name]
		g_1_vars = [var for var in t_vars if 'g_1_' in var.name]
		d_vars = [var for var in t_vars if 'd_' in var.name]

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			discrimitator_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_discriminator = discrimitator_optimizer.minimize(self.loss_cls, var_list = d_vars)

			g_0_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_g_0 = g_0_optimizer.minimize(self.loss_g_0, var_list = g_0_vars)

			g_1_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
			self.train_step_g_1 = g_1_optimizer.minimize(self.loss_g_1, var_list = g_1_vars)


	def __call__(self):
		seed = 9
		batch_size = 10
		start_images = 0
		number_of_images = 1000
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

			n_epochs = 1000

			for epoch in range(n_epochs):
				# TODO - this fails
				for i in range(int(np.floor(float(number_of_images*train_ratio)/batch_size))):
					f = open("processfile.txt", "w")
					f.write('processing image # ' + str(start_images + ((i*batch_size)+batch_size)) +' in batch ' +str(i) +' in epoch ' +str(epoch))
					f.close()

					batch_pos = data_pos['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
					batch_neg = data_neg['train_data'][(i*batch_size):((i*batch_size)+batch_size)]
					labels_pos = data_pos['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]
					labels_neg = data_neg['train_labels'][(i*batch_size):((i*batch_size)+batch_size)]

					# train discriminator
					loss, _ = self.sess.run(
						[self.loss_cls, self.train_step_discriminator],
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
					loss, _, image, image_1, image_2, image_3, image_4, loss_cls ,loss_per ,loss_pix_0 ,loss_pix_1 ,loss_dual ,loss_gan ,loss_g_0 ,loss_g_1  = self.sess.run(
						[self.loss_g_1, self.train_step_g_1, self.test_image,self.test_image_1,self.test_image_2,self.test_image_3,self.test_image_4, self.loss_cls, self.loss_per_0, self.loss_pix_0, self.loss_pix_1, self.loss_dual_0, self.loss_gan_0, self.loss_g_0, self.loss_g_1],
						feed_dict = {
							self.X_neg : batch_neg,
							self.X_pos : batch_pos,
							self.Y_neg : labels_neg,
							self.Y_pos : labels_pos
						}
					)
					print(f'Loss for g_1 is: {loss}')

					print(f'all them losses: {loss_cls} ,{loss_per} ,{loss_pix_0} ,{loss_pix_1} ,{loss_dual} ,{loss_gan} ,{loss_g_0} ,{loss_g_1}')

					im = Image.fromarray(scale_image(image).astype('uint8'))
					im.save('test.png')
					im.close()

					im = Image.fromarray(scale_image(image_1).astype('uint8'))
					im.save('test_res.png')
					im.close()

					im = Image.fromarray(scale_image(image_2).astype('uint8'))
					im.save('test_org.png')
					im.close()

					im = Image.fromarray(scale_image(image_3).astype('uint8'))
					im.save('test_res_remove.png')
					im.close()

					im = Image.fromarray(scale_image(image_4).astype('uint8'))
					im.save('test_remove.png')
					im.close()

					self.save(start_images + ((i*batch_size)+batch_size))

fg = FaceGAN()
fg()
