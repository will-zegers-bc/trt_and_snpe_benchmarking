from argparse import ArgumentParser
import functools
from collections import namedtuple
import logging
from math import ceil
import os
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf

from benchmarking_common import (ANNOTATION_DIR, 
                                 IMAGE_DATA_DIR,
                                 output_manager,
                                 process_input_file,
                                 tf_session_manager)
from model_meta import NETS, reverse_label_map_lookup

ImageData = namedtuple('ImageData', 'filename label')


def _get_directory_label(path):
    xml = os.listdir(path)[0]
    xml_path = os.path.join(path, xml)
    return next(ET.parse(xml_path).iter('name')).text.replace('_', ' ')


def _collect_test_files(image_dir, annotation_dir):
    logging.info("Collecting test files")

    test_files = []
    for data_dir, label_dir in zip(os.listdir(image_dir), os.listdir(annotation_dir)):
        assert data_dir == label_dir

        data_path = os.path.join(image_dir, data_dir)
        label_path = os.path.join(annotation_dir, label_dir)
        label = _get_directory_label(label_path)

        test_files.extend([
            ImageData(os.path.join(data_path, jpg), label) for jpg in os.listdir(data_path)
        ])

    return test_files


def load_test_set_files_and_labels(images_path, labels_path, size, seed=42):
    test_files = _collect_test_files(images_path, labels_path)
    np.random.seed(seed)
    np.random.shuffle(test_files)

    return test_files[:size]


def load_test_set_data(net_meta, batch_size=16):
    image_resizer = functools.partial(process_input_file, net_meta)
    test_set = ((map(image_resizer, [img.filename for img in test_files[i:i+batch_size]]),
                 (img.label for img in test_files[i:i+batch_size]))
                for i in range(0, len(test_files), batch_size))

    return test_set


def run_tf_accuracy_test(net_meta, test_files, batch_size):
        test_data = load_test_set_data(net_meta, batch_size)
        with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):
            predictions = []
            label_ids = []
            reverse_lookup = functools.partial(reverse_label_map_lookup, net_meta['num_classes'])
            for i, (batch, labels) in enumerate(test_data):
                logging.info("Processing batch %s of %s" % (i+1, int(ceil(len(test_files) / batch_size))))
                output = tf_sess.run([tf_output], feed_dict={
                    tf_input: batch
                })[0]
                predictions.extend(output.argmax(axis=1))

                label_ids.extend(map(reverse_lookup, labels))

            return predictions, label_ids


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_set_size', '-s', type=int, default=1024)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'pretty'])
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output) as output:
        if args.format == 'csv':
            output.write("net_name,throughput\n")
        
        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue

            test_files = load_test_set_files_and_labels(IMAGE_DATA_DIR, ANNOTATION_DIR, args.test_set_size)
            predictions, labels = run_tf_accuracy_test(net_meta, test_files, args.batch_size)
