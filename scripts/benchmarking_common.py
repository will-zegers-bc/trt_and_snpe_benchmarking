import contextlib
import sys

import cv2
import tensorflow as tf

IMAGE_BASE_DIR='data/images/'
IMAGE_DATA_DIR=IMAGE_BASE_DIR+'Images/'
ANNOTATION_DIR=IMAGE_BASE_DIR+'Annotation/'


def process_input_file(net_meta, img_file):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (net_meta['input_width'], net_meta['input_height']))

    return net_meta['preprocess_fn'](image)


@contextlib.contextmanager
def output_manager(output_file=None):
    try:
        if output_file is None:
            out = sys.stdout
        else:
            out = open(output_file, 'w')
        yield out
    finally:
        out.close()
        

@contextlib.contextmanager
def tf_session_manager(net_meta):
    with open(net_meta['frozen_graph_filename'], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    with tf.Session(config=tf_config, graph=graph) as tf_sess:
        tf_input = tf_sess.graph.get_tensor_by_name(net_meta['input_name'] + ':0')
        tf_output = tf_sess.graph.get_tensor_by_name(net_meta['output_names'][0] + ':0')

        yield tf_sess, tf_input, tf_output 
