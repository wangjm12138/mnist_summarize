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
tf.app.flags.DEFINE_string("EVAL_DATA","../data/mnist_validation.tfrecord",'test data')
tf.app.flags.DEFINE_integer("EVAL_STEPS",1000,'eval steps')
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

"""read predict data"""
dataset_predict = tf.data.TFRecordDataset(FLAGS.EVAL_DATA)
dataset_predict = dataset_predict.map(read_and_decode)
iterator_predict = dataset_predict.repeat().batch(FLAGS.EVAL_BATCH_SIZE).make_one_shot_iterator()
#iterator = dataset.batch(FLAGS.TRAIN_BATCH_SIZE).make_one_shot_iterator()
image_predict, label_predict = iterator_predict.get_next()

"""model"""
x = tf.placeholder(tf.float32, shape=[None,28,28,1])
y = tf.placeholder(tf.int64, shape=[None])
input_layer = tf.reshape(x, [-1, 28, 28, 1])
conv1 = tf.layers.conv2d(
	inputs=input_layer,
	filters=32,
	kernel_size=[5, 5],
	padding="same",
	activation=tf.nn.relu)
print(conv1)
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

logits = tf.layers.dense(inputs=dropout, units=10)


#train accuracy
classes = tf.argmax(logits,axis=1)
correct_prediction = tf.equal(classes, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""用save保存的只有.data(变量的值，没有结构),index,meta(原图，计算图的结构，没有变量值)"""
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"./output/model.ckpt"+'-'+str(100))
	print("Model restore.")
	image_p,labels_p= sess.run([image_predict,label_predict])
	accuracy_v= sess.run(accuracy,feed_dict={x:image_p,y:labels_p})
	print("predict_accuracy:{}".format(accuracy_v))
		
