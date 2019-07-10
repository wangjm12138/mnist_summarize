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

from tensorflow.python import pywrap_tensorflow

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


#reader = pywrap_tensorflow.NewCheckpointReader("./output/model.ckpt-100")
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#	print("tensor_name:",key)
#pdb.set_trace()
"""fetches和feed_dict的key 可以是name，https://github.com/tensorflow/tensorflow/issues/3378，https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph"""
"""用save保存的.data文件保存的是变量值，.index文件保存的是.data文件中数据和 .meta文件中结构图之间的对应关系"""
"""恢复图的方式有两种:
	一种保存源代码，这时候只需要导入.data就行，因为源代码的tensor会添加到默认图中，这时候.data的变量值直接导入，即可用
	另一种不想写原来代码，这时候需要导入.meta和.data，如果事先知道了input/ouput的tensor名称，即可根据tensor名称来feed_dict进去，如果不知道，先打印一下图的tensor名称,
	还有就是借助tensorboard来先看一下图的结构"""
with tf.Session() as sess:
	saver = tf.train.import_meta_graph("./output/model.ckpt-100.meta")
	saver.restore(sess,"./output/model.ckpt"+'-'+str(100))
	image_p,labels_p= sess.run([image_predict,label_predict])
	"""可以先打印一下图里面tensor的名称,如下面的，上面import_meta_graph已经把MetaGraphDef的protobuf转化为图了，再序列化成GraphDef,然后再打印出来node
	因为.meta是一种MetaGraphDef的protobuf格式，也可以看相应proto的文件然后根据定义的打印一下node,
	他们之间关系，GraphDef是一个protobuf定义一个类，其目的是为了定义图保存时的一种结构，包括tensor，操作,能保存常量，但是保存不了变量值
	结构如:
	node {
	  name: "Placeholder"     # 注释：这是一个叫做 "Placeholder" 的node
	  op: "Placeholder"
	  attr {
	    key: "dtype"
	    value {
	      type: DT_FLOAT
	    }
	  }
	  attr {
	    key: "shape"
	    value {
	      shape {
	        unknown_rank: true
	      }
	    }
	  }
	}
	node {
	  name: "mul"                 # 注释：一个 Mul（乘法）操作
	  op: "Mul"
	  input: "Placeholder"        # 使用上面的node（即Placeholder和Placeholder_1）
	  input: "Placeholder_1"      # 作为这个Node的输入
	  attr {
	    key: "T"
	    value {
	      type: DT_FLOAT
	    }
	  }
	}
	而MetaGraph是计算图和相关元数据构成的，也是由proto文件定义的一个类，里面有包含GraphDef，上面介绍的，总体如下：
	1.MetaInfoDef，存一些原信息(版本或者用户信息等) 
    2.GraphDef，MetaGraph核心内容之一
	3.SaverDef,图的save信息，比如同时保持的check-point数量，需要保存Tensor名字等
    4.CollectionDef任何需要特殊注意的Python对象。

    所以Graph,GraphDef,MetaGraph关系如下：
	MetaGraph是保存的图全部信息，而GraphDef只是MetaGraph是它的核心部分.
	而GraphDef恢复构建的图无法训练，因为没有保存任何变量信息，而meta是可以的，因为它保存了变量信息，
	但是没有实际值，所以一般.meta和.data是配套使用，如果只恢复.meta，会发现变量是随机值。
	Graph就是一个类了，并且可以序列化成GraphDef和MetaGraph.
	"""

	#a = [n.name for n in tf.get_default_graph().as_graph_def().node]
	#print(a)
	result = sess.run("v3/kernel:0",feed_dict={"v1:0":image_p,"v2:0":labels_p})
	print(result)
