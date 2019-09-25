# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

from argparse import ArgumentParser
import logging
import os
import subprocess

import uff

from model_meta import (NETS,
                        FROZEN_GRAPHS_DIR, 
                        CHECKPOINTS_DIR,
                        PLANS_DIR,
                        UFFS_DIR)

UFF_TO_PLAN_EXE_PATH = 'build/bin/uffToPlan'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_type', type=str, nargs='+', choices=['float', 'half'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    for data_type in args.data_type:
        if not os.path.exists(UFFS_DIR):
            os.makedirs(UFFS_DIR)

        plan_base_dir = os.path.join(PLANS_DIR, data_type)
        if not os.path.exists(plan_base_dir):
            os.makedirs(plan_base_dir)

        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue

            plan_filename = net_meta['plan_filename'].format(data_type)
            if not os.path.exists(net_meta['uff_filename']):
                uff_model = uff.from_tensorflow_frozen_model(
                    frozen_file=net_meta['frozen_graph_filename'],
                    output_nodes=net_meta['output_names'],
                    output_filename=net_meta['uff_filename'],
                    text=False,
                )
                if os.path.exists(plan_filename)
                    os.remove(plan_filename)

            if not os.path.exists(plan_filename):
                cmd_args = [
                    net_meta['uff_filename'],
                    plan_filename,
                    net_meta['input_name'],
                    str(net_meta['input_height']),
                    str(net_meta['input_width']),
                    net_meta['output_names'][0],
                    str(args.batch_size),
                    str(1 << 30),
                    data_type, # float / half
                ]
                subprocess.call([UFF_TO_PLAN_EXE_PATH] + cmd_args)
