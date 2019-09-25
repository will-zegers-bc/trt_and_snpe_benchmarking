from argparse import ArgumentParser
import logging
import os
import subprocess

from create_file_list import create_file_list
from create_raws import RESIZE_METHOD_BILINEAR, convert_img
from model_meta import DLCS_DIR, IMAGES_DIR, NETS, SAMPLES_DIR


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target_dsp', default=False, action='store_true')
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
                   '--graph', net_meta['frozen_graph_filename'],
                   '--input_dim', net_meta['input_name'], shape,
                   '--out_node', net_meta['output_names'][0],
                   '--dlc', net_meta['dlc_filename'],
                   '--allow_unconsumed_nodes']
            subprocess.call(cmd)

        if args.target_dsp:
            # TODO: need more quantization inputs
            raw_dir = os.path.join(IMAGES_DIR, 'raw', str(size))
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
                for img_file in os.listdir(SAMPLES_DIR):
                    img_path = os.path.join(SAMPLES_DIR, img_file)
                    raw_path = os.path.join(raw_dir, img_file)

                    convert_img(img_path, raw_path, size, RESIZE_METHOD_BILINEAR)

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
