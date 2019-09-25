# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import os

import numpy as np
import sys
sys.path.append("third_party/models/research/")
sys.path.append("third_party/models")
sys.path.append("third_party/")
sys.path.append("third_party/models/research/slim/")
import tensorflow.contrib.slim as tf_slim
import slim.nets as nets
import slim.nets.vgg
import slim.nets.inception
import slim.nets.resnet_v1
import slim.nets.resnet_v2
import slim.nets.mobilenet_v1
import slim.nets.mobilenet.mobilenet_v2

MOBILENET_V2_URL_BASE = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/'
MODELS_URL_BASE= 'http://download.tensorflow.org/models/'
CHECKPOINTS_DIR = 'data/checkpoints/'
FROZEN_GRAPHS_DIR = 'data/frozen_graphs/'
PLANS_DIR = 'data/plan/'
DLCS_DIR = 'data/dlc/'
UFFS_DIR = 'data/uff/'

IMAGES_DIR='data/images/'
INPUTS_DIR=os.path.join(IMAGES_DIR, 'Images')
LABELS_DIR=os.path.join(IMAGES_DIR, 'Annotation')
SAMPLES_DIR=os.path.join(IMAGES_DIR, 'samples')


def create_label_map(label_file='data/imagenet_labels_1001.txt'):
    label_map = {}
    with open(label_file, 'r') as f:
        labels = f.readlines()
        for i, label in enumerate(labels):
            label_map[i] = label.replace('\n', '')
    return label_map
        

IMAGENET2012_LABEL_MAP = create_label_map()

def reverse_label_map_lookup(num_classes, label):
    for k, v in IMAGENET2012_LABEL_MAP.items():
        if label in v:
            return k + num_classes - 1001;
    print("Invalid label found!: {}".format(label))
    return 1000

def preprocess_vgg(image):
    return np.array(image, dtype=np.float32) - np.array([123.68, 116.78, 103.94])

def postprocess_vgg(output):
    output = output.flatten()
    predictions_top5 = np.argsort(output)[::-1][0:5]
    labels_top5 = [IMAGENET2012_LABEL_MAP[p + 1] for p in predictions_top5]
    return labels_top5

def preprocess_inception(image):
    return 2.0 * (np.array(image, dtype=np.float32) / 255.0 - 0.5)

def postprocess_inception(output):
    output = output.flatten()
    predictions_top5 = np.argsort(output)[::-1][0:5]
    labels_top5 = [IMAGENET2012_LABEL_MAP[p] for p in predictions_top5]
    return labels_top5

def mobilenet_v1_1p0_224(*args, **kwargs):
    kwargs['depth_multiplier'] = 1.0
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)

def mobilenet_v1_0p5_160(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.5
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)

def mobilenet_v1_0p25_128(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.25
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)   

def mobilenet_v2_1p4_224_arg_scope(*args, **kwargs):
    kwargs['depth_multiplier'] = 1.4
    return nets.mobilenet.mobilenet_v2.mobilenet_base(*args, **kwargs)


NETS = {

    'vgg_16': {
        'url': MODELS_URL_BASE + 'vgg_16_2016_08_28.tar.gz',
        'model': nets.vgg.vgg_16,
        'arg_scope': nets.vgg.vgg_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_16/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'checkpoint_filename': CHECKPOINTS_DIR + 'vgg_16.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_16.pb',
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'vgg_16.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'vgg_16_quantized.dlc',
        'uff_filename': UFFS_DIR + 'vgg_16.uff',
        'plan_filename': PLANS_DIR + '{}/vgg_16.plan',
    },

    'vgg_19': {
        'url': MODELS_URL_BASE + 'vgg_19_2016_08_28.tar.gz',
        'model': nets.vgg.vgg_19,
        'arg_scope': nets.vgg.vgg_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_19/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'checkpoint_filename': CHECKPOINTS_DIR + 'vgg_19.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_19.pb',
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'vgg_19.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'vgg_19_quantized.dlc',
        'uff_filename': UFFS_DIR + 'vgg_19.uff',
        'plan_filename': PLANS_DIR + '{}/vgg_19.plan',
    },

    'inception_v1': {
        'url': MODELS_URL_BASE + 'inception_v1_2016_08_28.tar.gz',
        'model': nets.inception.inception_v1,
        'arg_scope': nets.inception.inception_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'inception_v1.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v1.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'inception_v1.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'inception_v1_quantized.dlc',
        'uff_filename': UFFS_DIR + 'inception_v1.uff',
        'plan_filename': PLANS_DIR + '{}/inception_v1.plan',
    },

    'inception_v2': {
        'exclude': True,
        'url': MODELS_URL_BASE + 'inception_v2_2016_08_28.tar.gz',
        'model': nets.inception.inception_v2,
        'arg_scope': nets.inception.inception_v2_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV2/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'inception_v2.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v2.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "bad results",
        'dlc_filename': DLCS_DIR + 'inception_v2.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'inception_v2_quantized.dlc',
        'uff_filename': UFFS_DIR + 'inception_v2.uff',
        'plan_filename': PLANS_DIR + '{}/inception_v2.plan',
    },

    'inception_v3': {
        'url': MODELS_URL_BASE + 'inception_v3_2016_08_28.tar.gz',
        'model': nets.inception.inception_v3,
        'arg_scope': nets.inception.inception_v3_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV3/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'inception_v3.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v3.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'inception_v3.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'inception_v3_quantized.dlc',
        'uff_filename': UFFS_DIR + 'inception_v3.uff',
        'plan_filename': PLANS_DIR + '{}/inception_v3.plan',
    },

    'inception_v4': {
        'url': MODELS_URL_BASE + 'inception_v4_2016_09_09.tar.gz',
        'model': nets.inception.inception_v4,
        'arg_scope': nets.inception.inception_v4_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV4/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'inception_v4.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v4.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'inception_v4.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'inception_v4_quantized.dlc',
        'uff_filename': UFFS_DIR + 'inception_v4.uff',
        'plan_filename': PLANS_DIR + '{}/inception_v4.plan',
    },
    
    'inception_resnet_v2': {
        'url': MODELS_URL_BASE + 'inception_resnet_v2_2016_08_30.tar.gz',
        'model': nets.inception.inception_resnet_v2,
        'arg_scope': nets.inception.inception_resnet_v2_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionResnetV2/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'inception_resnet_v2_2016_08_30.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_resnet_v2.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'dlc_filename': DLCS_DIR + 'inception_resnet_v2.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'inception_resnet_v2_quantized.dlc',
        'uff_filename': UFFS_DIR + 'inception_resnet_v2.uff',
        'plan_filename': PLANS_DIR + '{}/inception_resnet_v2.plan',
    },

    'resnet_v1_50': {
        'url': MODELS_URL_BASE + 'resnet_v1_50_2016_08_28.tar.gz',
        'model': nets.resnet_v1.resnet_v1_50,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v1_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_50.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'dlc_filename': DLCS_DIR + 'resnet_v1_50.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v1_50_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v1_50.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v1_50.plan',
    },

    'resnet_v1_101': {
        'url': MODELS_URL_BASE + 'resnet_v1_101_2016_08_28.tar.gz',
        'model': nets.resnet_v1.resnet_v1_101,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v1_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_101.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'dlc_filename': DLCS_DIR + 'resnet_v1_101.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v1_101_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v1_101.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v1_101.plan',
    },

    'resnet_v1_152': {
        'url': MODELS_URL_BASE + 'resnet_v1_152_2016_08_28.tar.gz',
        'model': nets.resnet_v1.resnet_v1_152,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v1_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_152.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'dlc_filename': DLCS_DIR + 'resnet_v1_152.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v1_152_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v1_152.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v1_152.plan',
    },

    'resnet_v2_50': {
        'url': MODELS_URL_BASE + 'resnet_v2_50_2017_04_14.tar.gz',
        'model': nets.resnet_v2.resnet_v2_50,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v2_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_50.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'resnet_v2_50.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v2_50_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v2_50.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v2_50.plan',
    },

    'resnet_v2_101': {
        'url': MODELS_URL_BASE + 'resnet_v2_101_2017_04_14.tar.gz',
        'model': nets.resnet_v2.resnet_v2_101,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v2_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_101.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'resnet_v2_101.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v2_101_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v2_101.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v2_101.plan',
    },

    'resnet_v2_152': {
        'url': MODELS_URL_BASE + 'resnet_v2_152_2017_04_14.tar.gz',
        'model': nets.resnet_v2.resnet_v2_152,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 'resnet_v2_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_152.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'resnet_v2_152.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'resnet_v2_152_quantized.dlc',
        'uff_filename': UFFS_DIR + 'resnet_v2_152.uff',
        'plan_filename': PLANS_DIR + '{}/resnet_v2_152.plan',
    },

    #'resnet_v2_200': {

    #},

    'mobilenet_v1_1p0_224': {
        'url': MODELS_URL_BASE + 'mobilenet_v1_1.0_224_2017_06_14.tar.gz',
        'model': mobilenet_v1_1p0_224,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 
            'mobilenet_v1_1.0_224.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_1p0_224.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'mobilenet_v1_1p0_224.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'mobilenet_v1_1p0_224_quantized.dlc',
        'uff_filename': UFFS_DIR + 'mobilenet_v1_1p0_224.uff',
        'plan_filename': PLANS_DIR + '{}/mobilenet_v1_1p0_224.plan',
    },

    'mobilenet_v1_0p5_160': {
        'url': MODELS_URL_BASE + 'mobilenet_v1_0.50_160_2017_06_14.tar.gz',
        'model': mobilenet_v1_0p5_160,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 160,
        'input_height': 160,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 
            'mobilenet_v1_0.50_160.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p5_160.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'mobilenet_v1_0p5_160.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'mobilenet_v1_0p5_160_quantized.dlc',
        'uff_filename': UFFS_DIR + 'mobilenet_v1_0p5_160.uff',
        'plan_filename': PLANS_DIR + '{}/mobilenet_v1_0p5_160.plan',
    },

    'mobilenet_v1_0p25_128': {
        'exclude': False,
        'url': MODELS_URL_BASE + 'mobilenet_v1_0.25_128_2017_06_14.tar.gz',
        'model': mobilenet_v1_0p25_128,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 128,
        'input_height': 128,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINTS_DIR + 
            'mobilenet_v1_0.25_128.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p25_128.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'mobilenet_v1_0p25_128.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'mobilenet_v1_0p25_128_quantized.dlc',
        'uff_filename': UFFS_DIR + 'mobilenet_v1_0p25_128.uff',
        'plan_filename': PLANS_DIR + '{}/mobilenet_v1_0p25_128.plan',
    },

    'mobilenet_v2_1p0_224': {
        'url': MOBILENET_V2_URL_BASE + 'mobilenet_v2_1.0_224.tgz',
        'model': nets.mobilenet.mobilenet_v2.mobilenet,
        'arg_scope': nets.mobilenet.mobilenet_v2.training_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['MobilenetV2/Predictions/Reshape_1'],
        'checkpoint_filename': CHECKPOINTS_DIR +
            'mobilenet_v2_1.0_224.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v2_1p0_224.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'mobilenet_v2_1p0_224.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'mobilenet_v2_1p0_224_quantized.dlc',
        'uff_filename': UFFS_DIR + 'mobilenet_v2_1p0_224.uff',
        'plan_filename': PLANS_DIR + '{}/mobilenet_v2_1p0_224.plan',
    },

    'mobilenet_v2_1p4_224': {
        'url': MOBILENET_V2_URL_BASE + 'mobilenet_v2_1.4_224.tgz',
        'model': nets.mobilenet.mobilenet_v2.mobilenet_v2_140,
        'arg_scope': nets.mobilenet.mobilenet_v2.training_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['MobilenetV2/Predictions/Reshape_1'],
        'checkpoint_filename': CHECKPOINTS_DIR +
            'mobilenet_v2_1.4_224.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v2_1p4_224.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'dlc_filename': DLCS_DIR + 'mobilenet_v2_1p4_224.dlc',
        'quantized_dlc_filename': DLCS_DIR + 'mobilenet_v2_1p4_224_quantized.dlc',
        'uff_filename': UFFS_DIR + 'mobilenet_v2_1p4_224.uff',
        'plan_filename': PLANS_DIR + '{}/mobilenet_v2_1p4_224.plan',
    },
}

