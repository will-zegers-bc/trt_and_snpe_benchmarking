# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

from argparse import ArgumentParser
import logging
from multiprocessing import Process
import os
import re
import subprocess
import time

import numpy as np

from benchmarking_common import (output_manager,
                                 preprocess_input_file,
                                 tf_session_manager,
                                 trt_engine_builder)
from model_meta import NETS, SAMPLES_DIR

TEST_IMAGE_PATH=os.path.join(SAMPLES_DIR, 'gordon_setter.jpg')


def record_memory_usage(pid, rate_hz=5):
    uss_kb, pss_kb, rss_kb = [], [], []
    while True:
        cmd = 'cat /proc/{}/smaps'.format(pid).split()                      
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = proc.communicate()                                    
        if err:                                                             
            logging.warn(err)
            break                                                           
        else:                                                               
            res = re.findall('Private_Dirty:\s*(\d+)\skB\n', output.decode('utf-8'))
            uss_kb.append(sum([int(n) for n in res]))
                                                                            
            res = re.findall('Pss:\s*(\d+)\skB\n', output.decode('utf-8'))
            pss_kb.append(sum([int(n) for n in res]))
                                                                            
            res = re.findall('Rss:\s*(\d+)\skB\n', output.decode('utf-8'))
            rss_kb.append(sum([int(n) for n in res]))

            if uss_kb[-1] == pss_kb[-1] == rss_kb[-1] == 0:  # process has died/exited
                break
                                                                            
        time.sleep(1./ rate_hz)                                             
    return uss_kb, pss_kb, rss_kb


def spin_trt_inferencing(net_meta, data_type, num_runs, test_image=TEST_IMAGE_PATH):
    engine = trt_engine_builder(net_meta, data_type)
    for i in range(num_runs):
        _ = engine.execute(test_image)


def spin_tf_inferencing(net_meta, num_runs=20, test_image=TEST_IMAGE_PATH):
    with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):

        shape = net_meta['input_width'], net_meta['input_height']
        image = preprocess_input_file(shape, net_meta['preprocess_fn'], test_image)

        # run network
        for i in range(num_runs):
            logging.info('Run %d of %d' % (i+1, num_runs))
            output = tf_sess.run([tf_output], feed_dict={
                tf_input: image[None, ...]
            })[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net_type', type=str, choices=['tf', 'trt'])
    parser.add_argument('--data_type', type=str, choices=['float', 'half'])
    parser.add_argument('--output_file', '-o', type=str, default=None)
    parser.add_argument('--num_runs', '-n', type=int, default=20)
    parser.add_argument('--test_image', '-i', type=str, default=TEST_IMAGE_PATH)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output_file) as output:
        output.write("net_name,peak_uss,peak_pss,peak_rss\n")
        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info('Skipping %s' % net_name)
                continue
        
            logging.info("Testing %s" % net_name)
            p = (Process(target=spin_tf_inferencing, args=(net_meta, args.num_runs))
                 if args.net_type == 'tf' else 
                 Process(target=spin_trt_inferencing, args=(net_meta, args.data_type, args.num_runs)))

            p.start()
            uss, pss, rss = record_memory_usage(p.pid)
            
            uss_max, pss_max, rss_max = map(np.max, [uss, pss, rss])
            csv_result = '{},{},{},{}\n'.format(net_name,uss_max,pss_max,rss_max)
            output.write(csv_result)
            logging.info(csv_result)
