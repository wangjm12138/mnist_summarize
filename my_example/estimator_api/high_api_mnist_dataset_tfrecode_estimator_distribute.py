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
tf.app.flags.DEFINE_string("EXPORT_NAME","mnist",'export name')
tf.app.flags.DEFINE_string("EVAL_NAME","mnist",'eval name')
tf.app.flags.DEFINE_string("MODEL_DIR","./output",'output dir folder')

tf.app.flags.DEFINE_integer("MAX_STEPS",1000,'train max steps')
tf.app.flags.DEFINE_integer("TRAIN_BATCH_SIZE",100,'tain batch size')
tf.app.flags.DEFINE_integer("EVAL_STEPS",100,'eval steps')
tf.app.flags.DEFINE_integer("EVAL_BATCH_SIZE",100,'eval batch size')

def read_and_decode(serialized_example):
	keys_to_features = {
	    'data': tf.FixedLenFeature([], tf.string, default_value=''),
	    'format': tf.FixedLenFeature([], tf.string, default_value='gray'),
	    'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
	}
	
	features = tf.parse_single_example(serialized_example, keys_to_features)
	# must same type to origin numpy which to tfrecode #
	image = tf.decode_raw(features['data'], tf.int8)
	image = tf.cast(image, tf.float32)
	#image = tf.reshape(image, [28, 28, 1],name="features_data_ph")
	image = tf.reshape(image, [28, 28, 1])
	#image = tf.image.per_image_standardization(image)
	label = features['label']
	#label = tf.one_hot(label, 10, 1, 0)
	return image, label


def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=0, batch_size=200, num_parallel_calls=None, prefetch_buffer_size=None):
	dataset = tf.data.TFRecordDataset(filenames)
	dataset = dataset.map(read_and_decode).repeat().batch(batch_size)
	#iterator = dataset.repeat().batch(batch_size).make_one_shot_iterator()
	#image, label = iterator.get_next()
	#return {'x':image},label
	#return image,label
	return dataset

def train_input():
	return input_fn(FLAGS.TRAIN_DATA,batch_size=FLAGS.TRAIN_BATCH_SIZE)

def eval_input():
	return input_fn(FLAGS.EVAL_DATA,batch_size=FLAGS.EVAL_BATCH_SIZE)

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# MNIST images are 28x28 pixels, and have one color channel
	print(features['x'].name)
	input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
	
	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 28, 1]
	# Output Tensor Shape: [batch_size, 28, 28, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28, 28, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	
	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 14, 14, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14, 14, 64]
	# Output Tensor Shape: [batch_size, 7, 7, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7, 7, 64]
	# Output Tensor Shape: [batch_size, 7 * 7 * 64]
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])#*****
	
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7 * 7 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	
	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 10]
	logits = tf.layers.dense(inputs=dropout, units=10)

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

	"""当导出函数用FinalExporter用于hook的train_accuracy_op,对训练打印的精度，要放在mode=tf.estimator.ModeKeys.PREDICT后面,因为在最后保存模型时候，调用的是self._model_fn(features=features, **kwargs)
		即这时候只有features，没有labels，而这时候return在mode=tf.estimator.ModeKeys.PREDICT这个条件,若果用estimator的export_save_model,可以自己选ModeKeys"""
	train_accuracy_op = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], labels), tf.float32),name="run_train_accuracy")

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


#SERVING_FUNCTIONS = {
#	'JSON': json_serving_input_fn,
#	'EXAMPLE': example_serving_input_fn,
#	'CSV': csv_serving_input_fn
#	}
#def csv_serving_input_fn():
#	"""Build the serving inputs."""
#	csv_row = tf.placeholder(shape=[None], dtype=tf.string)
#	features = _decode_csv(csv_row)
#	features.pop(constants.LABEL_COLUMN)
#	return tf.estimator.export.ServingInputReceiver(features,
#													{'csv_row': csv_row})

#def example_serving_input_fn():
#	"""Build the serving inputs."""
#	example_bytestring = tf.placeholder(
#		shape=[None],
#		dtype=tf.string,
#	)
#	features = tf.parse_example(
#		example_bytestring,
#		tf.feature_column.make_parse_example_spec(featurizer.INPUT_COLUMNS))
#	return tf.estimator.export.ServingInputReceiver(
#		features, {'example_proto': example_bytestring})

# [START serving-function]
"""tf.estimator.export.ServingInputReceiver(features, receiver_tensors),这里features指的就是model_fn的形式，
也就是input_fn最终输出的(features,labels)，要跟这里一致,而receiver_tensors,是指外接给tensorflow的形式。
即:
features是模型的格式
receiver_tensors是外接给程序格式
receiver_tensors->features这个过程就是这个函数在做，即json_serving_input_fn，或者注释csv_serving_input_fn,
example_serving_input_fn这些函数做的事情。
"""
def json_serving_input_fn():
	"""Build the serving inputs."""
	inputs = {}
	x = tf.placeholder(tf.float32, shape=[None,28,28,1])
	inputs['x'] = x
	return tf.estimator.export.ServingInputReceiver(inputs,inputs)

	#return tf.estimator.export.TensorServingInputReceiver(x,x)
	"""https://github.com/tensorflow/tensorflow/issues/11674,关于要是我datase没有以字典形式输入，这边会默认是{feature:x}会出错"""
	"""这时候要用TensorServingInputReceiver,这个可以接受单个tensor，而不用以字典形式"""

def main(unused_argv):

	# Train the model
	train_spec = tf.estimator.TrainSpec(train_input, max_steps=FLAGS.MAX_STEPS, hooks=[logging_hook])

	eval_spec = tf.estimator.EvalSpec(eval_input, steps=FLAGS.EVAL_STEPS, name=FLAGS.EVAL_NAME)
	"""
	estimator使用分布式时候，input_fn必须返回的是dataset的类型，	
	不管tf.distribute.MirroredStrategy(devices=["/cpu"])或者填tf.distribute.MirroredStrategy(devices=["/cpu:0","/cpu:1"])都是默认用所有的CPU来做分布式
	当指定GPU时候可以指定GPU数量
	tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
	"""
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu"])
	run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var(),save_checkpoints_steps=100,train_distribute=mirrored_strategy)
	run_config = run_config.replace(model_dir=FLAGS.MODEL_DIR)

	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, config=run_config)

	tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
	#mnist_classifier.export_saved_model(export_dir_base="./output/mnist", serving_input_receiver_fn = json_serving_input_fn)

if __name__ == "__main__":
	tf.app.run()
