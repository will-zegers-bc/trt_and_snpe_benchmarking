import os
import tarfile
import urllib

import tensorflow as tf
import tensorflow.contrib.slim as tf_slim

from convert_relu6 import convertRelu6
from model_meta import (CHECKPOINTS_DIR,
                        FROZEN_GRAPHS_DIR,
                        IMAGES_DIR,
                        INPUTS_DIR,
                        LABELS_DIR,
                        NETS,
                        SAMPLES_DIR)

INPUTS_URL='http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
LABELS_URL='http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'

SAMPLE_IMAGES={
    'gordon_setter': 'http://farm3.static.flickr.com/2017/2496831224_221cd963a2.jpg',
    'lifeboat': 'http://farm3.static.flickr.com/2582/4106642219_190bf0f817.jpg',
    'golden_retriever': 'http://farm4.static.flickr.com/3226/2719028129_9aa2e27675.jpg',
}


def download_and_extract(url, dest_dir):
    tar_file = url[-url[::-1].find('/'):]
    tar_path = os.path.join(dest_dir, tar_file)

    urllib.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=dest_dir)
    os.remove(tar_path)


def convert_model_to_frozen_graph(net_meta):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as tf_sess:
        tf_sess = tf.Session(config=tf_config)
        tf_input = tf.placeholder(
            tf.float32, 
            (
                None, 
                net_meta['input_height'], 
                net_meta['input_width'], 
                net_meta['input_channels']
            ),
            name=net_meta['input_name']
        )

        with tf_slim.arg_scope(net_meta['arg_scope']()):
            tf_net, tf_end_points = net_meta['model'](
                tf_input, 
                is_training=False,
                num_classes=net_meta['num_classes']
            )

        tf_saver = tf.train.Saver()
        tf_saver.restore(
            save_path=net_meta['checkpoint_filename'], 
            sess=tf_sess
        )
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            tf_sess,
            tf_sess.graph_def,
            output_node_names=net_meta['output_names']
        )

        frozen_graph = convertRelu6(frozen_graph)

        with open(net_meta['frozen_graph_filename'], 'wb') as f:
            f.write(frozen_graph.SerializeToString())


if __name__ == '__main__':
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)

    if not os.path.exists(FROZEN_GRAPHS_DIR):
        os.makedirs(FROZEN_GRAPHS_DIR)

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude']:
            continue

        if not os.path.exists(net_meta['checkpoint_filename']):
            print('[+] Downloading and extracting {} checkpoint'.format(net_name))
            download_and_extract(net_meta['url'], CHECKPOINTS_DIR)

        if not os.path.exists(net_meta['frozen_graph_filename']):
            print('[+] Converting {} to a TF frozen graph'.format(net_name))
            convert_model_to_frozen_graph(net_meta)

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    if not os.path.exists(INPUTS_DIR):
        print('[+] Downloading Stanford Dogs dataset')
        download_and_extract(INPUTS_URL, IMAGES_DIR)
        download_and_extract(LABELS_URL, IMAGES_DIR)

    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

    for name, url in SAMPLE_IMAGES.items():
        image_jpg = name + '.jpg'
        image_path = os.path.join(SAMPLES_DIR, image_jpg)
        if not os.path.exists(image_path):
            print('[+] Downloading sample image {}'.format(image_jpg))
            urllib.urlretrieve(url, image_path)
