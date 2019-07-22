import tensorflow as tf
from PIL import Image
import pdb

class Read(object):
 
    def __init__(self, file_dir, batch_size):
        self.FILE_DIR = file_dir
        self.BATCH_SIZE = batch_size
 
    def _read_and_decode(self, serialized_example):
        keys_to_features = {
            'data': tf.FixedLenFeature([], tf.string, default_value=''),
            'format': tf.FixedLenFeature([], tf.string, default_value='gray'),
            'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }
 
        features = tf.parse_single_example(serialized_example, keys_to_features)
        print(features['data'])
        image = tf.decode_raw(features['data'], tf.int8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28, 28, 1])
        #image = tf.image.per_image_standardization(image)
        label = features['label']
        #label = tf.one_hot(label, 10, 1, 0)
        return image, label
 
    def tfr_reader(self, min_after_dequeue=1000):
        files = tf.train.match_filenames_once([self.FILE_DIR])
        filename_queue = tf.train.string_input_producer(files, shuffle=True)
 
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        image, label = self._read_and_decode(serialized_example)
 
        capacity = min_after_dequeue + 3 * self.BATCH_SIZE
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.BATCH_SIZE, num_threads=1,
                                                          capacity=capacity, min_after_dequeue=min_after_dequeue)
        return image_batch, label_batch
 
    def tfr_data(self, shuffle=True):
        data = tf.data.TFRecordDataset(self.FILE_DIR)
        data = data.map(self._read_and_decode)
 
        iterator = data.batch(1).make_one_shot_iterator()
        image, label = iterator.get_next()
        return image, label 

filenames = "./mnist_train1.tfrecord"

a=Read(filenames,100)
image,label = a.tfr_data()

with tf.Session() as sess:
    feature = sess.run([image,label])
    print(feature[0],feature[1])
