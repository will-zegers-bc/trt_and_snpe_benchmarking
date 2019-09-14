# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

from argparse import ArgumentParser
import logging
from multiprocessing import Process
import re
import subprocess
import time

import cv2
import numpy as np
import tensorflow as tf

from benchmarking_common import output_manager, tf_session_manager
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR

TEST_IMAGE_PATH='data/images/gordon_setter.jpg'


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


def test_memory_footprint(net_meta, num_runs=20, test_image=TEST_IMAGE_PATH):
    with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):

        # load and preprocess image
        image = cv2.imread(TEST_IMAGE_PATH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (net_meta['input_width'], net_meta['input_height']))
        image = net_meta['preprocess_fn'](image)

        # run network
        for i in range(num_runs):
            logging.info('Run %d of %d' % (i, num_runs))
            output = tf_sess.run([tf_output], feed_dict={
                tf_input: image[None, ...]
            })[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_runs', '-n', type=int, default=20)
    parser.add_argument('--test_image', '-i', type=str, default=TEST_IMAGE_PATH)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'pretty'])
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output) as output:
        if args.format == 'csv':
            output.write("net_name,average_uss,max_uss,average_pss,max_pss,average_rss,max_rss")
        for net_name, net_meta in NETS.keys():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info('Skipping %s' % net_name)
                continue
    
            p = Process(target=test_memory_footprint, args=(net_meta, args.num_runs))
            p.start()
            uss, pss, rss = record_memory_usage(p.pid)
            
            uss_max, pss_max, rss_max = map(np.max, [uss, pss, rss])
            if args.format == 'csv':
                res = '{},{},{},{}'.format(
                    net_name,uss_max,pss_max,rss_max)
            else:
                res = '{}\t{} KB\t{} KB\t{} KB'.format(
                    net_name,uss_max,pss_max,rss_max)
            output.write(output)
