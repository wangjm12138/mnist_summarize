#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
import multiprocessing
starttime = datetime.datetime.now()
tf.logging.set_verbosity(tf.logging.INFO)
import pdb

FLAGS = tf.app.flags.FLAGS

TRAIN_DATA=["../data/train-00000-of-01024","../data/train-00001-of-01024","../data/train-00002-of-01024","../data/train-00003-of-01024","../data/train-00004-of-01024","../data/train-00005-of-01024","../data/train-00006-of-01024","../data/train-00007-of-01024"]


#EVAL_DATA=["/opt/ai/input/data/validation/validation-00000-of-00128"]

#TRAIN_DATA=["/home/wangjm2/imagenet-data/train-00000-of-01024","/home/wangjm2/imagenet-data/train-00001-of-01024"]

#tf.app.flags.DEFINE_string("MODEL_DIR","/opt/ai/output",'output dir folder')
tf.app.flags.DEFINE_string("MODEL_DIR","./output",'output dir folder')
tf.app.flags.DEFINE_integer("MAX_STEPS",10000,'train max steps')
tf.app.flags.DEFINE_integer("EVAL_STEPS",1000,'eval steps')

tf.app.flags.DEFINE_string("EXPORT_NAME","imagenet",'export name')
tf.app.flags.DEFINE_string("EVAL_NAME","imagenet",'eval name')

tf.app.flags.DEFINE_integer("TRAIN_BATCH_SIZE",100,'tain batch size')
tf.app.flags.DEFINE_integer("EVAL_BATCH_SIZE",100,'eval batch size')

def read_and_decode(serialized_example):

	keys_to_features = {
		'image/height': tf.FixedLenFeature([], tf.int64, default_value=0),
		'image/width': tf.FixedLenFeature([], tf.int64, default_value=0),
		'image/colorspace': tf.FixedLenFeature([], tf.string, default_value='RGB'),
		'image/channels': tf.FixedLenFeature([], tf.int64, default_value=3),
		'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
		'image/class/text': tf.FixedLenFeature([], tf.string, default_value=''),
		'image/format': tf.FixedLenFeature([], tf.string, default_value='JPEG'),
		'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
		'image/encoded': tf.FixedLenFeature([], tf.string, default_value='')
	}
	features = tf.parse_single_example(serialized_example, keys_to_features)
	height = features['image/height']
	width = features['image/width']
	colorspace = features['image/colorspace']
	channels = features['image/channels']
	text = features['image/class/text']
	Format = features['image/format']
	filename = features['image/filename']
	label = features['image/class/label']
	#decode_jpeg function will  return image type is uint8 ,shape [height,width,channels]
	image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
	#image = tf.image.resize_images(image,(32,32))
	image = tf.image.resize_images(image,(224,224))
	#image = tf.reshape(image,[32,32,3])
	image = tf.reshape(image,[224,224,3])
	image = tf.image.per_image_standardization(image)
	#label = tf.one_hot(label, 10, 1, 0)

	return image, label


def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=0, batch_size=200, num_parallel_calls=None, prefetch_buffer_size=None):
	dataset = tf.data.TFRecordDataset(filenames)
	num_parallel_calls = multiprocessing.cpu_count()
	dataset = dataset.map(read_and_decode,num_parallel_calls=8)
	#dataset = dataset.map(read_and_decode)
	#iterator = dataset.repeat().batch(batch_size).make_one_shot_iterator()
	#iterator = dataset.batch(batch_size).make_one_shot_iterator()
	#image, label = iterator.get_next()
	#return image,label
	return dataset.repeat(6).batch(batch_size).prefetch(buffer_size=batch_size)
	#return dataset.repeat(6).batch(batch_size)
	#return dataset.prefetch(buffer_size=batch_size)
	#return dataset

def train_input():
	return input_fn(TRAIN_DATA,batch_size=FLAGS.TRAIN_BATCH_SIZE)

def eval_input():
	return input_fn(EVAL_DATA,batch_size=FLAGS.EVAL_BATCH_SIZE)

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	print(features)
	print(labels)
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# MNIST images are 28x28 pixels, and have one color channel
	input_layer = tf.reshape(features, [-1, 224, 224, 3])
	
	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 224, 224, 3]
	# Output Tensor Shape: [batch_size, 224, 224, 64]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv2 = tf.layers.conv2d(
		inputs=conv1,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 224, 224, 64]
	# Output Tensor Shape: [batch_size, 112, 112, 64]
	pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2,2])
	
	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 112, 112, 64]
	# Output Tensor Shape: [batch_size, 56, 56, 128]
	conv3 = tf.layers.conv2d(
		inputs=pool1,
		filters=128,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=128,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 112, 112, 128]
	# Output Tensor Shape: [batch_size, 56, 56, 128]
	pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[2,2])

	conv5 = tf.layers.conv2d(
		inputs=pool2,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv6 = tf.layers.conv2d(
		inputs=conv5,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv7 = tf.layers.conv2d(
		inputs=conv6,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	# Pooling Layer #3
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 56, 56, 256]
	# Output Tensor Shape: [batch_size, 28, 28, 256]

	pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=[2,2])
	conv8 = tf.layers.conv2d(
		inputs=pool3,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	conv9 = tf.layers.conv2d(
		inputs=conv8,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	conv10 = tf.layers.conv2d(
		inputs=conv9,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	# Pooling Layer #4
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28, 28, 512]
	# Output Tensor Shape: [batch_size, 14, 14, 512]

	pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=[2,2])
	conv11 = tf.layers.conv2d(
		inputs=pool4,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	conv12 = tf.layers.conv2d(
		inputs=conv11,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	conv13 = tf.layers.conv2d(
		inputs=conv12,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	# Pooling Layer #5
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14, 14, 512]
	# Output Tensor Shape: [batch_size, 7, 7, 512]

	pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=[2,2])

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7, 7, 512]
	# Output Tensor Shape: [batch_size, 7 * 7 * 512]
	pool2_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])#*****
	
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7 * 7 * 512]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)
	
	# Add dropout operation; 0.5 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 4096]
	# Output Tensor Shape: [batch_size, 1001]
	#logits = tf.layers.dense(inputs=dropout, units=10)
	logits = tf.layers.dense(inputs=dropout, units=1001)
	print(dropout)
	print(logits)
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)#oneshot , number_lable

  # Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])} 
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _get_session_config_from_env_var():
	tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
	
	if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
		'index' in tf_config['task']):
	    # Master should only communicate with itself and ps
		if tf_config['task']['type'] == 'master':
			return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
	    # Worker should only communicate with itself and ps
		elif tf_config['task']['type'] == 'worker':
			return tf.ConfigProto(device_filters=[
				'/job:ps',
				'/job:worker/task:%d' % tf_config['task']['index']
			])
	return None


def main(unused_argv):
	
	# Train the model
	#train_spec = tf.estimator.TrainSpec(train_input, max_steps=FLAGS.MAX_STEPS)
	#eval_spec = tf.estimator.EvalSpec(eval_input, steps=FLAGS.EVAL_STEPS, name=FLAGS.EVAL_NAME)
	#run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var(),save_checkpoints_secs=None,save_checkpoints_steps=None)
	#run_config = tf.estimator.RunConfig(save_checkpoints_secs=None,save_summary_steps=None)
#	distribute_strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=2)
	#run_config = tf.estimator.RunConfig(train_distribute=distribute_strategy,save_checkpoints_secs=None,save_checkpoints_steps=None)
	run_config = tf.estimator.RunConfig(save_checkpoints_secs=None,save_checkpoints_steps=None,model_dir='/tmp/wangjmss')

	imagenet_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, config=run_config)



	imagenet_classifier.train(train_input,max_steps=FLAGS.MAX_STEPS)
	#tf.estimator.train_and_evaluate(imagenet_classifier, train_spec, eval_spec)
	#tf.estimator.train_and_evaluate(imagenet_classifier, train_spec, None)


	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	endtime = datetime.datetime.now()
	last_time=(endtime-starttime).seconds
	tf.logging.info(last_time)
	#os.system("echo %s > /opt/ai/output/a.txt"%str(last_time))
	#time.sleep(30)
	#print(eval_results)

if __name__ == "__main__":
	tf.logging.info("start")
	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	tf.app.run()
