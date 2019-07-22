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

"""
a = tf.random_uniform(shape=[1,2],dtype=tf.int32,minval=1,maxval=2)
with tf.Session() as sess:
	b = sess.run(a)
	#c = sess.run(a,feed_dict={a:[[2,2]]})
像比如这样sess.run的fetchs，可以是tf.Operation，tf.Tensor,tf.SparaseTensor,可以是string，
像上面这样，b=[[1,1]],而且c的这句其实就是把a这个tensor的值重新赋上[[2,2]]
"""

######################################################################################

"""第一种模式，以tensor作为dataset的输入,可以看到
一个列表包括两个元素，一个是a，一个是b，维度都是一行两列，比如a=[[1,1]],b=[[2,2]]
batch最多只能2，当然大于2只会设置为2,不设置会取第一个元素
但是有个地方要注意，batch这个函数会在数据前面加一个维度表示取的数据量，即注释#1和#2
如果用注释#1，得到的结果是：[[2 2]]，就是a的值，维度是shape[1,2]
而用注释#2，得到的结果是：[[[2,2]]]，它的维度是shape[1,1,2],可以看到batch这个函数加了一维
这边还要注意不能用开启#3配合#4来用make_one_shot_iterator，因为tf.random_uniform内部自带了隐含
的变量，没有初始化，所以用make_one_shot_iterator会报错,如果a,b都是tf.constant就可以，如第二段程序
所以iterator.initalizer包含了变量的初始化
"""
"""
a = tf.random_uniform(shape=[1,2],dtype=tf.int32,minval=1,maxval=2)
b = tf.random_uniform(shape=[1,2],dtype=tf.int32,minval=1,maxval=2)

dataset = tf.data.Dataset.from_tensor_slices([a,b])
dataset = dataset.map(lambda x:x+1)
#iterator = dataset.make_initializable_iterator()   #1
iterator = dataset.batch(1).make_initializable_iterator() #2
#iterator = dataset.make_one_shot_iterator() #3
num = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer) #4
	feature = sess.run(num)
	print(feature)
"""
######################################################################################
"""
a = tf.constant([[1,1]])
b = tf.constant([[3,3]])

dataset = tf.data.Dataset.from_tensor_slices([a,b])
dataset = dataset.map(lambda x:x+1)
iterator = dataset.make_one_shot_iterator() #3
num = iterator.get_next()

with tf.Session() as sess:
	feature = sess.run(num)
"""

######################################################################################
"""第二种模式，以tensor作为dataset的输入"""

def add(x):
	print("in")
	return x+1

#a = map(add,[1,1,1,1])
#print(list(a))

dataset = tf.data.Dataset.range(10)
dataset = dataset.map(add)
iterator = dataset.batch(2).make_one_shot_iterator() #3
num = iterator.get_next()

with tf.Session() as sess:
	feature = sess.run(num)
	print(feature)

