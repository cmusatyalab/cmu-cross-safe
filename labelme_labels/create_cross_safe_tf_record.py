# Copyright 2019 Carnegie Mellon University
#
# This is based on the following file:
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py
# The original file is licensed under the Apache License, Version 2.0 with
# the following copyright notice:
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import hashlib
import io
import os
import random

import contextlib2
import PIL.Image
import tensorflow as tf
import pickle
import logging

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from lxml import etree

# from https://github.com/agschwender/pilbox/issues/34#issuecomment-84093912
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None

flags = tf.app.flags
flags.DEFINE_string('label_dir', '', 'Directory with labels')
flags.DEFINE_string('image_dir', '', 'Directory with images')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'cross_safe_label_map.pbtxt',
                    'Path to label map proto')

FLAGS = flags.FLAGS

CLASSES = {
    'Don\'t walk' : 1,
    'dont walk' : 1,
    'walk' : 2,
    'Walk' : 2,
    'countdown' : 3,
    'off' : 4,
}

OTHER = 'other'
TRAIN_FILENAME = 'train.record'
VAL_FILENAME = 'val.record'
TRAIN_PICKLE_FILENAME = 'train_files.p'
VAL_PICKLE_FILENAME = 'val_files.p'
TEST_PICKLE_FILENAME = 'test_files.p'


def filename_to_tf_example(filename, label_dir, image_dir, category_index):
    print(filename)
    img_path = os.path.join(image_dir, '{}.JPEG'.format(filename))
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        print(image.format)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image.size

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        classes = []
        classes_text = []

        label_path = os.path.join(
            label_dir, '{}.xml'.format(filename))
        with open(label_path) as label_file:
            xml = etree.fromstring(label_file.read())
            for child in xml.findall('object'):
                label_from_file = child.find('name').text
                class_number = CLASSES.get(label_from_file)
                if class_number is None:
                    if label_from_file == OTHER:
                        logging.info('Skipping {} label'.format(OTHER))
                    else:
                        raise Exception('Bad Label {}'.format(label_from_file))
                else:
                    classes.append(class_number)

                    category = category_index[class_number]['name']
                    classes_text.append(category.encode('utf8'))

                    bndbox = child.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    xmins.append(xmin / width)
                    ymins.append(ymin / height)
                    xmaxs.append(xmax / width)
                    ymaxs.append(ymax / height)

            feature_dict = {
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            return example

    raise Exception('Error creating example')


def create_tf_record(
        output_filename, category_index, label_dir, image_dir, files):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_record = tf_record_close_stack.enter_context(
            tf.python_io.TFRecordWriter(output_filename))
        for filename in files:
            tf_example = filename_to_tf_example(
                filename, label_dir, image_dir, category_index)
            output_record.write(tf_example.SerializeToString())


def main(_):
    logging.basicConfig(level=logging.INFO)

    label_dir = FLAGS.label_dir
    image_dir = FLAGS.image_dir
    category_index = label_map_util.create_category_index_from_labelmap(
        FLAGS.label_map_path)

    label_filenames = set()
    for filename in os.listdir(FLAGS.label_dir):
        if filename.endswith('.xml'):
            label_filenames.add(filename[:-4])
        elif filename == '.gitattributes':
            logging.info('skipping gitattributes')
        else:
            print('Bad filename', filename)

    image_filenames = set()
    for filename in os.listdir(FLAGS.image_dir):
        if filename.endswith('.JPEG'):
            image_filenames.add(filename[:-5])
        elif filename == '.gitattributes':
            logging.info('skipping gitattributes')
        else:
            print('Bad filename', filename)

    print('Number of text files', len(label_filenames))
    print('Number of images', len(image_filenames))
    print('Difference',
          image_filenames.difference(label_filenames))

    files = list(label_filenames)
    random.shuffle(files)

    # Remove 20% of data
    files = files[:int(len(files) * 0.8)]

    num_examples = len(files)
    num_train_val = int(0.8 * num_examples)
    num_val = int(0.2 * num_train_val)
    num_train = num_train_val - num_val

    val_start = num_train
    test_start = num_train + num_val

    train_files = files[:val_start]
    val_files = files[val_start:test_start]
    test_files = files[test_start:]

    train_record = os.path.join(FLAGS.output_dir, TRAIN_FILENAME)
    val_record = os.path.join(FLAGS.output_dir, VAL_FILENAME)
    
    create_tf_record(
        train_record, category_index, label_dir, image_dir, train_files)
    create_tf_record(
        val_record, category_index, label_dir, image_dir, val_files)

    train_pickle_file = os.path.join(
        FLAGS.output_dir, TRAIN_PICKLE_FILENAME)
    with open(train_pickle_file, 'wb') as f:
        pickle.dump(train_files, f)

    val_pickle_file = os.path.join(
        FLAGS.output_dir, VAL_PICKLE_FILENAME)        
    with open(val_pickle_file, 'wb') as f:
        pickle.dump(val_files, f)

    test_pickle_file = os.path.join(
        FLAGS.output_dir, TEST_PICKLE_FILENAME)
    with open(test_pickle_file, 'wb') as f:
        pickle.dump(test_files, f)

    print('num train', len(train_files))
    print('num val', len(val_files))
    print('num test', len(test_files))


if __name__ == '__main__':
    tf.app.run()
