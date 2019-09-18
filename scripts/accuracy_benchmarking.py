from __future__ import division

from argparse import ArgumentParser
import functools
import logging
from math import ceil
import os
from xml.etree import ElementTree as ET

import numpy as np

from benchmarking_common import (output_manager,
                                 preprocess_input_file,
                                 tf_session_manager,
                                 trt_engine_builder)
from model_meta import INPUTS_DIR, LABELS_DIR, NETS, reverse_label_map_lookup


def _get_directory_label(path, num_classes):
    xml = os.listdir(path)[0]
    xml_path = os.path.join(path, xml)

    label_str = next(ET.parse(xml_path).iter('name')).text.replace('_', ' ')
    return reverse_label_map_lookup(num_classes, label_str)


def _collect_test_files(image_dir, annotation_dir, num_classes):
    logging.info("Collecting test files")

    file_paths, labels = [], []
    for data_dir, label_dir in zip(map(sorted, map(os.listdir, (image_dir, annotation_dir)))):
        assert data_dir == label_dir, "Data/label mismatch: data_dir: %s, label_dir: %s" % (data_dir, label_dir)
        data_path = os.path.join(image_dir, data_dir)
        label_path = os.path.join(annotation_dir, label_dir)

        label = _get_directory_label(label_path, num_classes)
        dir_contents = os.listdir(data_path)

        labels.extend([label] * len(dir_contents))
        file_paths.extend([
            os.path.join(data_path, file_) for file_ in dir_contents
        ])

    return file_paths, labels


def load_test_set_files_and_labels(images_path, labels_path, size, num_classes, seed=42):
    np.random.seed(seed)

    files, labels = _collect_test_files(images_path, labels_path, num_classes)
    selected = np.random.permutation(len(files))[:size]

    return [files[s] for s in selected], [labels[s] for s in selected]


def load_test_set_data(net_meta, files, batch_size):
    shape = net_meta['input_width'], net_meta['input_height']
    image_processor = functools.partial(preprocess_input_file, shape, net_meta['preprocess_fn'])
    image_data = (map(image_processor, [img for img in files[i:i+batch_size]])
                  for i in range(0, len(files), batch_size))

    return image_data


def run_trt_accuracy_test(net_meta, data_type, files, batch_size):
    predictions = []
    engine = trt_engine_builder(net_meta, data_type)
    for img in files:
        # TODO: fix execute interface to match inputs and outputs of TF session.run
        output = np.array(engine.execute(img)).reshape(1, -1)
        predictions.extend(output.argmax(axis=1))

    return predictions


def run_tf_accuracy_test(net_meta, files, batch_size=1):
        predictions = []
        images = load_test_set_data(net_meta, files, batch_size)
        with tf_session_manager(net_meta) as (tf_sess, tf_input, tf_output):
            for i, batch in enumerate(images):
                logging.info("Processing batch %s of %s" % (i+1, int(ceil(len(files) / batch_size))))
                output = tf_sess.run([tf_output], feed_dict={
                    tf_input: batch
                })[0]
                predictions.extend(output.argmax(axis=1))

        return predictions


def calculate_class_precision_and_recall(predictions, label, num_classes):
    precisions, recalls = np.empty((2, num_classes))
    for label in set(labels):
        tp = sum(pr == lb == label for pr, lb in zip(predictions, labels))
        if not tp:
            precision = recall = 0
        else:
            fp = sum(pr == label and lb != label for pr, lb in zip(predictions, labels))
            fn = sum(pr != label and lb == label for pr, lb in zip(predictions, labels))

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        precisions[label], recalls[label] = precision, recall
    return precisions, recalls


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net_type', type=str, choices=['tf', 'trt'])
    parser.add_argument('--data_type', type=str, choices=['float', 'half'])
    parser.add_argument('--output_file', '-o', type=str, default=None)
    parser.add_argument('--test_set_size', '-s', type=int, default=1024)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with output_manager(args.output_file) as output:
        output.write('net_name,accuracy,precision,recall\n')
        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue

            files, labels = load_test_set_files_and_labels(INPUTS_DIR,
                                                           LABELS_DIR,
                                                           args.test_set_size,
                                                           net_meta['num_classes'])

            logging.info("Testing %s" % net_name)
            predictions = (run_tf_accuracy_test(net_meta, files, args.batch_size)
                           if args.net_type == 'tf' else
                           run_trt_accuracy_test(net_meta, args.data_type, files, args.batch_size))

            accuracy = np.equal(predictions, labels).mean()
            precision, recall = calculate_class_precision_and_recall(predictions, labels, net_meta['num_classes'])

            avg_precision = precision[np.unique(labels)].mean()
            avg_recall = recall[np.unique(labels)].mean()

            csv_result = '{},{},{},{}\n'.format(net_name, accuracy, avg_precision, avg_recall)
            output.write(csv_result)
            logging.info(csv_result)
