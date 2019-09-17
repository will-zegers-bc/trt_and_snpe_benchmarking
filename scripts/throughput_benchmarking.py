# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
from __future__ import division

from argparse import ArgumentParser
import logging
import os
import time

import numpy as np
import tensorflow as tf

from benchmarking_common import (output_manager,
                                 preprocess_input_file,
                                 tf_session_manager)
from model_meta import CHECKPOINT_DIR, FROZEN_GRAPHS_DIR, NETS, PLAN_DIR
from tensor_rt import InferenceEngine, NetConfig

TEST_IMAGE_PATH='data/images/gordon_setter.jpg'


def test_trt_average_throughput(net_meta, data_type, num_runs=50, test_image=TEST_IMAGE_PATH):
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
    engine = InferenceEngine(net_config)

    avg_latency = engine.measure_throughput(test_image, num_runs)
    return 1 / avg_latency


def test_average_throughput(net_meta, num_runs=50, test_image=TEST_IMAGE_PATH):
    with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):

        shape = net_meta['input_width'], net_meta['input_height']
        image = process_input_file(shape, net_meta['preprocess_fn'], test_image)

        # run network
        times = [0.] * (num_runs+1)
        for i in range(num_runs + 1):
            t0 = time.time()
            output = tf_sess.run([tf_output], feed_dict={
                tf_input: image[None, ...]
            })[0]
            times[i] = time.time() - t0
        return 1 / np.mean(times[1:]) # don't include first run


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net_type', type=str, choices=['tf', 'trt'])
    parser.add_argument('--data_type', type=str, choices=['half', 'float'])
    parser.add_argument('--output_file', '-o', type=str, default=None)
    parser.add_argument('--num_runs', '-c', type=int, default=50)
    parser.add_argument('--test_image', '-i', type=str, default=TEST_IMAGE_PATH)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output_file) as output:
        output.write("net_name,throughput\n")
        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue
    
            logging.info('Testing %s' % net_name)
            avg_throughput = (test_average_throughput(net_meta, args.num_runs)
                              if args.net_type == 'tf' else
                              test_trt_average_throughput(net_meta, args.data_type, args.num_runs))
    
            output.write('{},{}\n'.format(net_name, avg_throughput))
