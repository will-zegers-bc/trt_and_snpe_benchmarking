import contextlib
import os
import sys

import cv2
import tensorflow as tf

from model_meta import PLAN_DIR
from tensor_rt import InferenceEngine, NetConfig

IMAGE_BASE_DIR='data/images/'
IMAGE_DATA_DIR=IMAGE_BASE_DIR+'Images/'
ANNOTATION_DIR=IMAGE_BASE_DIR+'Annotation/'


def preprocess_input_file(shape, preprocess_fn, img_file):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape)

    return preprocess_fn(image)


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


def trt_engine_builder(net_meta, data_type):
    plan_dir = os.path.join(PLAN_DIR, data_type)
    net_config = NetConfig(
        plan_path=os.path.join(plan_dir, net_meta['plan_filename']),
        input_node_name=net_meta['input_name'],
        output_node_name=net_meta['output_names'][0],
        preprocess_fn_name=net_meta['preprocess_fn'].__name__,
        input_height=net_meta['input_height'],
        input_width=net_meta['input_width'],
        num_output_categories=net_meta['num_classes'],
        max_batch_size=1)

    return InferenceEngine(net_config)
