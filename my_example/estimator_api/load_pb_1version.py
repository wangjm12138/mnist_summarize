import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
import json
import pdb
#output_graph_path = "/home/wangjm2/docker/v100/mnist/tfrecode/aws/mnist_git/my_example/estimator_api/output/mnist/1562320453/"
output_graph_path = "/home/wangjm2/docker/v100/mnist/tfrecode/aws/mnist_git/my_example/estimator_api/output/mnist/"
dir_name = os.listdir(output_graph_path)
output_graph_path = os.path.join(output_graph_path,dir_name[0])
print(output_graph_path)

filenames = "../data/mnist_validation.tfrecord"
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

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(read_and_decode)
iterator = dataset.batch(1).make_one_shot_iterator()
image, label = iterator.get_next()

with tf.Session() as sess:
	feature = sess.run([image,label])
	print(feature[0],feature[1])
"""saved_model_cli show --dir output/mnist/1562320453/ --all 可以看到模型的全部，包括signature，还有name"""
graph2 = tf.Graph()
with graph2.as_default():
	with tf.Session(graph=tf.Graph()) as sess:
		_ = loader.load(sess, [tag_constants.SERVING], output_graph_path)
		input_x = sess.graph.get_tensor_by_name('Placeholder:0')
		op = sess.graph.get_tensor_by_name('ArgMax:0')
		ret = sess.run(op,  feed_dict={input_x:feature[0]})
		print(ret)



