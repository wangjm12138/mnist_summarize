#coding:utf-8
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import datetime
import json
import os
import pdb

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("TRAIN_DATA","../data/mnist_train.tfrecord",'input data')
tf.app.flags.DEFINE_string("EVAL_DATA","../data/mnist_validation.tfrecord",'test data')
tf.app.flags.DEFINE_integer("MAX_STEPS",1000,'train max steps')
tf.app.flags.DEFINE_integer("EVAL_STEPS",1000,'eval steps')
tf.app.flags.DEFINE_integer("TRAIN_BATCH_SIZE",100,'tain batch size')
tf.app.flags.DEFINE_integer("EVAL_BATCH_SIZE",100,'eval batch size')

def read_and_decode(serialized_example):
	keys_to_features = {
	    'data': tf.FixedLenFeature([], tf.string, default_value=''),
	    'format': tf.FixedLenFeature([], tf.string, default_value='gray'),
	    'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
	}
	
	features = tf.parse_single_example(serialized_example, keys_to_features)
	image = tf.decode_raw(features['data'], tf.int8)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, [28, 28, 1])
	#image = tf.image.per_image_standardization(image)
	label = features['label']
	#label = tf.one_hot(label, 10, 1, 0)
	return image, label


"""read train data"""
dataset_train = tf.data.TFRecordDataset(FLAGS.TRAIN_DATA)
dataset_train = dataset_train.map(read_and_decode)
iterator_train = dataset_train.repeat().batch(FLAGS.TRAIN_BATCH_SIZE).make_one_shot_iterator()
#iterator = dataset.batch(FLAGS.TRAIN_BATCH_SIZE).make_one_shot_iterator()
image, label = iterator_train.get_next()
#print(image,label)
"""read test data"""
dataset_eval = tf.data.TFRecordDataset(FLAGS.EVAL_DATA)
dataset_eval = dataset_eval.map(read_and_decode)
iterator_eval = dataset_eval.repeat().batch(FLAGS.EVAL_BATCH_SIZE).make_one_shot_iterator()
#iterator = dataset.batch(FLAGS.TRAIN_BATCH_SIZE).make_one_shot_iterator()
image_test, label_test = iterator_eval.get_next()

"""model"""
x = tf.placeholder(tf.float32, shape=[None,28,28,1],name="v1")
y = tf.placeholder(tf.int64, shape=[None],name="v2")
input_layer = tf.reshape(x, [-1, 28, 28, 1])
conv1 = tf.layers.conv2d(
	inputs=input_layer,
	filters=32,
	kernel_size=[5, 5],
	padding="same",
	activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(
	inputs=pool1,
	filters=64,
	kernel_size=[5, 5],
	padding="same",
	activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)

logits = tf.layers.dense(inputs=dropout, units=10,name="v3")


"""train accuracy, api = tf.equal + tf.reduce_mean = tf.metrics.accuracy"""
classes = tf.argmax(logits,axis=1)
#correct_prediction = tf.equal(classes, y)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""tf.metrics.accuracy返回两个值，accuracy为到上一个batch为止的准确度，update_op为更新本批次后的准确度"""
accuracy = tf.metrics.accuracy(labels=y, predictions=classes)

#probabilities = tf.nn.softmax(logits, name="softmax_tensor")

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss=loss)

saver = tf.train.Saver()
#print(tf.GraphKeys.GLOBAL_VARIABLES)
#print(tf.global_variables())
#pdb.set_trace()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#if use tf.metrics.accuracy is must have
	sess.run(tf.local_variables_initializer())
	for i in range(6000):
		##get data
		image_t,labels_t= sess.run([image,label])
		##calculate train op,loss,accuracy
		_, loss_value,accuracy_w= sess.run([train_op,loss,accuracy],feed_dict={x:image_t,y:labels_t})
		#pdb.set_trace()
		"""print every 100 step train accuracy and test accuracy"""
		if i%100 == 0:
			image_v,labels_v= sess.run([image_test,label_test])
			accuracy_v= sess.run(accuracy,feed_dict={x:image_v,y:labels_v})
			print("num:{},Loss:{:4f},train_accuracy:{},test_accuracy:{}".format(i,loss_value,accuracy_w,accuracy_v))
			save_path  = saver.save(sess,"./output/model.ckpt",global_step=i)
			print("num:%s, model saved in path: %s" % (i,save_path))
	
