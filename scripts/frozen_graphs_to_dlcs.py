import os
import subprocess

from create_file_list import create_file_list
from create_raws import RESIZE_METHOD_BILINEAR, convert_img
from model_meta import DLCS_DIR, IMAGES_DIR, NETS


if __name__ == '__main__':
    if not os.path.exists(DLCS_DIR):
        os.makedirs(DLCS_DIR)

    for net_name, net_meta in NETS.items():
        size = str(net_meta['input_width'])
        shape = '1,{},{},{}'.format(
            net_meta['input_width'],
            net_meta['input_height'],
            net_meta['input_channels']
        )

        print('[+] Converting ' + net_name +' to SNPE DLC format')
        cmd = ['snpe-tensorflow-to-dlc',
               '--graph', net_meta['frozen_graph_filename'],
               '--input_dim', net_meta['input_name'], shape,
               '--out_node', net_meta['output_names'][0],
               '--dlc', os.path.join(DLCS_DIR, net_meta['dlc_filename']),
               '--allow_unconsumed_nodes']
        subprocess.call(cmd)

        raw_dir = os.path.join(IMAGES_DIR, 'raw', size)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            for img_file in os.listdir('data/images/samples'):
                img_path = os.path.join('data/images/samples', img_file)
                raw_path = os.path.join(raw_dir, img_file)

                convert_img(img_path, raw_path, size, RESIZE_METHOD_BILINEAR)

        txt_name = 'data/inputs_raw_list_' + size + '.txt'
        if not os.path.exists(txt_name):
            create_file_list(raw_dir, txt_name, '*.raw')

        print('[+] Creating ' + net_name + ' quantized model')
        cmd = ['snpe-dlc-quantize',
               '--input_dlc', os.path.join(DLCS_DIR, net_meta['dlc_filename']),
               '--input_list', os.path.abspath(txt_name),
               '--output_dlc', os.path.join(DLCS_DIR, net_meta['quantized_dlc_filename'])]
        subprocess.call(cmd)
