# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
from __future__ import division

from argparse import ArgumentParser
import logging
import os
import time

import numpy as np

from benchmarking_common import (output_manager,
                                 preprocess_input_file,
                                 snpe_engine_builder,
                                 tf_session_manager,
                                 trt_engine_builder)
from model_meta import NETS, SAMPLES_DIR

TEST_IMAGE_PATH=os.path.join(SAMPLES_DIR, 'gordon_setter.jpg')


def test_snpe_average_throughput(neta_meta, runtime='cpu', num_runs=50, test_image=TEST_IMAGE_PATH):
    if not net_meta['snpe_supported'][runtime]:
        return float('nan')

    shape = net_meta['input_width'], net_meta['input_height']
    image = preprocess_input_file(shape, net_meta['preprocess_fn'], test_image)

    engine = (snpe_engine_builder(net_meta['quantized_dlc_filename'], runtime)
              if runtime == 'dsp' else
              snpe_engine_builder(net_meta['dlc_filename'], runtime))

    times = [0.] * (num_runs+1)
    for i in range(num_runs + 1):
        t0 = time.time()
        engine.execute(image)
        times[i] = time.time() - t0
    return 1 / np.mean(times[1:])  # don't include first run


def test_trt_average_throughput(net_meta, data_type, num_runs=50, test_image=TEST_IMAGE_PATH):
    shape = net_meta['input_width'], net_meta['input_height']
    image = preprocess_input_file(shape, net_meta['preprocess_fn'], test_image)

    engine = trt_engine_builder(net_meta, data_type)
    times = [0.] * (num_runs+1)
    for i in range(num_runs + 1):
        t0 = time.time()
        engine.execute(image)
        times[i] = time.time() - t0
    return 1 / np.mean(times[1:])  # don't include first run


def test_tf_average_throughput(net_meta, num_runs=50, test_image=TEST_IMAGE_PATH):
    with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):
        shape = net_meta['input_width'], net_meta['input_height']
        image = preprocess_input_file(shape, net_meta['preprocess_fn'], test_image)

        times = [0.] * (num_runs+1)
        for i in range(num_runs + 1):
            t0 = time.time()
            output = tf_sess.run([tf_output], feed_dict={
                tf_input: image[None, ...]
            })[0]
            times[i] = time.time() - t0
        return 1 / np.mean(times[1:])  # don't include first run


if __name__ == '__main__':
    parser = ArgumentParser()

    # Common
    parser.add_argument('net_type', type=str, choices=['snpe', 'tf', 'trt'])
    parser.add_argument('--output_file', '-o', type=str, default=None)
    parser.add_argument('--num_runs', '-c', type=int, default=50)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')

    # TRT
    parser.add_argument('--data_type', type=str, choices=['half', 'float'])

    # SNPE
    parser.add_argument('--runtime', type=str, choices=['cpu', 'dsp', 'gpu'], default='cpu')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output_file, 'w+') as output:
        output.write("net_name,throughput\n")

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            logging.info("Skipping {}".format(net_name))
            continue
    
        logging.info('Testing %s' % net_name)
        throughput = (test_tf_average_throughput(net_meta, args.num_runs)
                      if args.net_type == 'tf' else
                      test_trt_average_throughput(net_meta, args.data_type, args.num_runs)
                      if args.net_type == 'trt' else
                      test_snpe_average_throughput(net_meta, args.runtime, args.num_runs))
    
        csv_result = '{},{}'.format(net_name, throughput)
        logging.info(csv_result)
        with output_manager(args.output_file, 'a') as output:
            output.write(csv_result+'\n')
