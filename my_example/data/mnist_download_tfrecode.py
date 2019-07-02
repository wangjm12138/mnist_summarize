# encoding:utf-8
"""
========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""
import os
import numpy
import struct
import requests
import pdb
import gzip
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot

host = "http://yann.lecun.com/exdb/mnist"
## 训练集文件
train_images_gz = 'train-images-idx3-ubyte.gz'
# 训练集标签文件
train_labels_gz = 'train-labels-idx1-ubyte.gz'
# 测试集文件
test_images_gz = 't10k-images-idx3-ubyte.gz'
# 测试集标签文件
test_labels_gz = 't10k-labels-idx1-ubyte.gz'

file_list = [train_images_gz,train_labels_gz,test_images_gz,test_labels_gz]

def download_mnist():
	global file_list
	for item in file_list: 
		if os.path.exists(item):
			print("Down load %s is already exists."%item)
		else:
			url = os.path.join(host,item)
			print("Down load {file} from {url}.".format(file=item,url=url))
			try:
				response = requests.get(url, stream=True)
				with open(item,"wb") as handle:
					for data in tqdm(response.iter_content()):
						handle.write(data)
			except Exception as e:
				raise Exception(e)

def _read32(bytestream):
	dt = numpy.dtype(numpy.uint32).newbyteorder('>')
	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, filename))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		return data


def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = numpy.arange(num_labels) * num_classes
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot
	
def extract_labels(filename, one_hot=False):
	"""Extract the labels into a 1D uint8 numpy array [index]."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, filename))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)
		if one_hot:
			return dense_to_one_hot(labels)
		return labels


def _int64_feature(value):
	if not isinstance(value, (tuple, list)):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _image_to_tfexample(image_data, image_format, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': _bytes_feature(image_data),
        'format': _bytes_feature(image_format),
        'label': _int64_feature(class_id)
    }))

def _add_to_tfrecord(images, labels, tfrecord_writer):
	with tf.Graph().as_default():
		for j in range(len(images)):
			#print(images[j],images[j].shape)
			"""shape(28,28,1) - > shape(28,28)"""
			image = numpy.squeeze(images[j])
			#print("=======")
			#print(image,image.shape)
			image = Image.fromarray(image)
			image = image.resize((28,28))
			image = image.tobytes()
			label = labels[j]
			#print(label)
			#pdb.set_trace()
			example = _image_to_tfexample(image, b'gray', label)
			tfrecord_writer.write(example.SerializeToString())
	
download_mnist()
"""shape (60000, 28, 28, 1) uint8 """
train_images = extract_images(train_images_gz)
"""shape (60000,) uint8 """
train_labels = extract_labels(train_labels_gz)
"""shape (10000, 28, 28, 1) uint8 """
test_images = extract_images(test_images_gz)
"""shape (10000,) uint8 """
test_labels = extract_labels(test_labels_gz)
print(len(train_images),len(train_labels),train_images.shape,train_labels.shape)
print(len(test_images),len(test_labels),test_images.shape,test_labels.shape)
"""show first images"""
#first_image = numpy.array(train_images[0],dtype='uint8')
#pixels = first_image.reshape((28,28))
#pyplot.imshow(pixels,cmap='gray')
#pyplot.savefig('./first.png')


"""to tfrecode fomat"""
training_filename = 'mnist_train.tfrecord'
testing_filename = 'mnist_validation.tfrecord'

# First, process the training data:
with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
	_add_to_tfrecord(train_images, train_labels, tfrecord_writer)

# Next, process the testing data:
with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
	_add_to_tfrecord(test_images, test_labels, tfrecord_writer)
















"""
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

:param idx_ubyte_file: idx文件路径
:return: n*row*col维np.array对象，n为图片数量
"""
"""
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

:param idx_ubyte_file: idx文件路径
:return: n*1维np.array对象，n为图片数量
"""
"""
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

:param idx_ubyte_file: idx文件路径
:return: n*row*col维np.array对象，n为图片数量
"""
"""
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

:param idx_ubyte_file: idx文件路径
:return: n*1维np.array对象，n为图片数量
"""

