import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
import json

output_graph_path = "/home/wangjm2/docker/v100/mnist/tfrecode/aws/mnist_git/my_example/estimator_api/output/export/mnist/1562234637"

#sess = tf.Session()
#with open(output_graph_def,'rb') as f:
#	graph_def = tf.GraphDef()
#	graph_def.ParseFromString(f.read())
#	sess.graph.as_default()
#	tf.import_graph_def(graph_def,name='') 
#
#
#sess.run(tf.global_variables_initializer())
#
#print(sess.run())



with tf.Session(graph=tf.Graph()) as sess:
	_ = loader.load(sess, [tag_constants.SERVING], output_graph_path)
	print(sess.graph.get_collection('inputs'))
	#input_x = sess.graph.get_tensor_by_name('x:0')
	#input_alias_map = json.loads(sess.graph.get_collection('inputs'))
	#print(input_alias_map)
	#sess.run(tf.global_variables_initializer())
 
    #input_x = sess.graph.get_tensor_by_name('x:0')
    #input_y = sess.graph.get_tensor_by_name('y:0')
 
    #op = sess.graph.get_tensor_by_name('op_to_store:0')
 
    #ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
    #print(ret)
