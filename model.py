import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras import layers
from generator import Generator
from discriminator import Discriminator
from utils import get_data
from tensorflow.keras.callbacks import TensorBoard as tb

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

seed = 9
batch_size = 100
number_of_images = 1000

data_neg, data_pos = get_data(number_of_images, 0.7, seed)

reset_graph(seed)

g_1_input = tf.placeholder(tf.float32, shape = [None, 128, 128, 3])
g_1_labels = tf.placeholder(tf.uint8, shape = [None, 1])

g_0_input = tf.placeholder(tf.float32, shape = [None, 128, 128, 3])
g_0_labels = tf.placeholder(tf.uint8, shape = [None, 1])

d_input = tf.placeholder(tf.float32, shape = [None, 128, 128, 3])
d_labels = tf.placeholder(tf.uint8, shape = [None, 1])

cls_loss = tf.placeholder(tf.float32)

g_0 = Generator(seed, g_0_input)
g_1 = Generator(seed, g_1_input)

r_0 = g_0.output
r_1 = g_1.output

x_tilde_0 = tf.add(r_0, g_0_input)
x_tilde_1 = tf.add(r_1, g_1_input)

d = Discriminator(seed, d_input)

# Discriminator loss
loss_cls = -tf.log(tf.reduce_sum(d.output))

optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4)
train_step = optimizer.minimize(cls_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	n_epochs = 10
	sess.run(init)
	for epoch in range(n_epochs):
		for i in range(int(np.floor(number_of_images/batch_size))):
			batch_pos = data_pos['train_data'][i*batch_size:(i*batch_size)+batch_size]
			batch_neg = data_neg['train_data'][i*batch_size:(i*batch_size)+batch_size]
			labels_pos = data_pos['train_labels'][i*batch_size:(i*batch_size)+batch_size]
			labels_neg = data_neg['train_labels'][i*batch_size:(i*batch_size)+batch_size]
			batch = np.concatenate((batch_pos, batch_neg), axis = 0)
			batch_labels = np.concatenate((labels_pos, labels_neg), axis = 0)

			tilde = sess.run(x_tilde_0, feed_dict = {
				g_0_input : batch_neg
			})

			first_d_loss = sess.run(loss_cls, feed_dict = {
				d_input: tilde
			})

			tilde = sess.run(x_tilde_1, feed_dict = {
				g_1_input : batch_pos
			})

			second_d_loss = sess.run(loss_cls, feed_dict = {
				d_input: tilde
			})

			third_d_loss = sess.run(loss_cls, feed_dict = {
				d_input: batch_pos
			})

			fourth_d_loss = sess.run(loss_cls, feed_dict = {
				d_input: batch_neg
			})

			total_cls_loss = first_d_loss, second_d_loss, third_d_loss, fourth_d_loss

			sess.run(train_step, feed_dict = {
				cls_loss: total_cls_loss
			})
			print(first_d_loss)
			"""sess.run(train_step, feed_dict = {
				X_pos : batch_pos,
				X_pos : batch_neg
			})

			print("Cost:", sess.run(loss_cls, feed_dict = {
				X_pos : batch_pos,
				X_pos : batch_neg
			})"""

