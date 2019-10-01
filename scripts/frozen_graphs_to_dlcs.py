from argparse import ArgumentParser
import logging
import os
import subprocess

from benchmarking_common import load_test_set_files_and_labels, preprocess_input_file
from create_file_list import create_file_list
from model_meta import DLCS_DIR, IMAGES_DIR, NETS, INPUTS_DIR, LABELS_DIR


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--quantize', '-q', default=False, action='store_true')
    parser.add_argument('--use_enhanced_quantizer', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)    

    if not os.path.exists(DLCS_DIR):
        os.makedirs(DLCS_DIR)

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            logging.info("Skipping {}".format(net_name))
            continue

        size = net_meta['input_width']
        shape = '1,{},{},{}'.format(
            net_meta['input_width'],
            net_meta['input_height'],
            net_meta['input_channels']
        )

        if not os.path.exists(net_meta['dlc_filename']):
            logging.info('[+] Converting ' + net_name +' to SNPE DLC format')
            cmd = ['snpe-tensorflow-to-dlc',
                   '--input_network', net_meta['frozen_graph_filename'],
                   '--input_dim', net_meta['input_name'], shape,
                   '--out_node', net_meta['output_names'][0],
                   '-o', net_meta['dlc_filename'],
                   '--allow_unconsumed_nodes']
            subprocess.call(cmd)

        if args.quantize:
            raw_dir = os.path.join(IMAGES_DIR, 'raw', str(size))
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
                image_files, _ = load_test_set_files_and_labels(INPUTS_DIR, 
                                                                LABELS_DIR,
                                                                20,
                                                                net_meta['num_classes'],
                                                                seed=555)
                for path in image_files:
                    filename = path[-path[::-1].find('/'):]
                    raw_path = os.path.join(raw_dir, filename.replace('jpg', 'raw'))

                    shape = net_meta['input_width'], net_meta['input_height']
                    raw_image = preprocess_input_file(shape, net_meta['preprocess_fn'], path)
                    raw_image.tofile(raw_path)

            txt_path = os.path.join(raw_dir, 'raw_list.txt')
            if not os.path.exists(txt_path):
                create_file_list(raw_dir, txt_path, '*.raw')

            if not os.path.exists(net_meta['quantized_dlc_filename']):
                logging.info('[+] Creating ' + net_name + ' quantized model')
                cmd = ['snpe-dlc-quantize',
                       '--input_dlc', net_meta['dlc_filename'],
                       '--input_list', txt_path,
                       '--output_dlc', net_meta['quantized_dlc_filename']]
                if args.use_enhanced_quantizer:
                    cmd.append('--use_enhanced_quantizer')
                subprocess.call(cmd)
