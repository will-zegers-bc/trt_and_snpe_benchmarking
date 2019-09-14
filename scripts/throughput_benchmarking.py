# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
from __future__ import division

from argparse import ArgumentParser
import logging
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

from benchmarking_common import output_manager, tf_session_manager
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR

TEST_IMAGE_PATH='data/images/gordon_setter.jpg'


def test_average_throughput(net_meta, num_runs=50, test_image=TEST_IMAGE_PATH):
    with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):

        # load and preprocess image
        image = cv2.imread(test_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (net_meta['input_width'], net_meta['input_height']))
        image = net_meta['preprocess_fn'](image)

        # run network
        times = [0.] * (num_runs+1)
        for i in range(args.num_runs + 1):
            t0 = time.time()
            output = tf_sess.run([tf_output], feed_dict={
                tf_input: image[None, ...]
            })[0]
            times[i] = time.time() - t0
        return 1 / np.mean(times[1:]) # don't include first run


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_runs', '-n', type=int, default=50)
    parser.add_argument('--test_image', '-i', type=str, default=TEST_IMAGE_PATH)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'pretty'])
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output) as output:
        if args.format == 'csv':
            output.write("net_name,throughput\n")
        
        for net_name, net_meta in NETS.items():
#        for net_name, net_meta in [('resnet_v1_50', NETS['resnet_v1_50'])]:
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue

            logging.info('Testing %s' % net_name)
            avg_throughput = test_average_throughput(net_meta, args.num_runs)
            if args.format == 'csv':
                output.write("{},{}\n".format(net_name, avg_throughput))
            else:
                output.write("{} @ {} inferences/sec\n".format(net_name, avg_throughput))
