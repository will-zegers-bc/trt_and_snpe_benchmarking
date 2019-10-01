import contextlib
import os
import sys
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf

from model_meta import reverse_label_map_lookup

def preprocess_input_file(shape, preprocess_fn, img_file, save_path=''):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape)

    if save_path:
        cv2.imwrite(save_path, image)
    return preprocess_fn(image)


def _get_directory_label(path, num_classes):
    xml = os.listdir(path)[0]
    xml_path = os.path.join(path, xml)

    label_str = next(ET.parse(xml_path).iter('name')).text.replace('_', ' ')
    return reverse_label_map_lookup(num_classes, label_str)


def _collect_test_files(image_dir, annotation_dir, num_classes):
    file_paths, labels = [], []
    for data_dir, label_dir in zip(*map(sorted, map(os.listdir, (image_dir, annotation_dir)))):
        assert data_dir == label_dir, "Data/label mismatch: data_dir: %s, label_dir: %s" % (data_dir, label_dir)
        data_path = os.path.join(image_dir, data_dir)
        label_path = os.path.join(annotation_dir, label_dir)

        label = _get_directory_label(label_path, num_classes)
        dir_contents = os.listdir(data_path)

        labels.extend([label] * len(dir_contents))
        file_paths.extend([
            os.path.join(data_path, file_) for file_ in dir_contents
        ])

    return file_paths, labels


def load_test_set_files_and_labels(images_path, labels_path, size, num_classes, seed=42):
    np.random.seed(seed)

    files, labels = _collect_test_files(images_path, labels_path, num_classes)
    selected = np.random.permutation(len(files))[:size]

    return [files[s] for s in selected], [labels[s] for s in selected]


@contextlib.contextmanager
def output_manager(output_file=None, mode='w'):
    out = None
    try:
        if output_file is None:
            out = sys.stdout
        else:
            out = open(output_file, mode)
        yield out
    finally:
        if out is not None and out is not sys.stdout:
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
    import PyTensorRT

    net_config = PyTensorRT.NetConfig(
        plan_path=net_meta['plan_filename'].format(data_type),
        input_node_name=net_meta['input_name'],
        output_node_name=net_meta['output_names'][0],
        input_height=net_meta['input_height'],
        input_width=net_meta['input_width'],
        input_channels=3,
        num_output_categories=net_meta['num_classes'],
        max_batch_size=1)

    return PyTensorRT.InferenceEngine(net_config)


def snpe_engine_builder(dlc_file, runtime):
    import PySNPE
    return PySNPE.InferenceEngine(dlc_file, runtime)

