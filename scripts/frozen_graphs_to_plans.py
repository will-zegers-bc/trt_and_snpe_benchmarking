# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

from argparse import ArgumentParser
import logging
import os

import uff

from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINTS_DIR, PLANS_DIR
from convert_plan import frozenToPlan


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_type', type=str, choices=['float', 'half'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    output_dir = os.path.join(PLAN_DIR, args.data_type)
    if not os.path.exists(output_dir):
        if not os.path.exists(PLAN_DIR):
            os.makedirs(PLAN_DIR)
        os.makedirs(output_dir)

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            logging.info("Skipping {}".format(net_name)) 
            continue

        logging.info("Convertings %s to PLAN" % net_name)
        frozenToPlan(
            net_meta['frozen_graph_filename'],
            net_meta['plan_filename'].format(args.data_type),
            net_meta['input_name'],
            net_meta['input_height'],
            net_meta['input_width'],
            net_meta['output_names'][0],
            args.batch_size, # batch size
            1 << 20, # workspace size
            args.data_type # data type
        )
