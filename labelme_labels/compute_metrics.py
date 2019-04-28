# Based on:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


import numpy as np
import tensorflow as tf
import pickle
import time
import heapq
import os
import logging

from distutils.version import StrictVersion
from PIL import Image
from lxml import etree
from types import SimpleNamespace
from sklearn.metrics import average_precision_score

# from https://github.com/agschwender/pilbox/issues/34#issuecomment-84093912
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None

# From https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#L632
MIN_SCORE_THRESH = 0.5
JACCARD_THRESHOLD = 0.5

IMG_DIR = '/home/ubuntu/cross_safe_april_2019/images'
LABEL_DIR = '/home/ubuntu/cross_safe_april_2019/labels'
CLASSES = {
  'Don\'t walk' : 1,
  'dont walk' : 1,
  'walk' : 2,
  'Walk' : 2,
  'countdown' : 3,
  'off' : 4,
}
OTHER = 'other'

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def jaccard(box_a, box_b):
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def load_ground_truths(filename, width, height):
  ground_truths = []
  label_path = os.path.join(LABEL_DIR, '{}.xml'.format(filename))
  with open(label_path) as label_file:
    xml = etree.fromstring(label_file.read())
    
    for child in xml.findall('object'):
      label_from_file = child.find('name').text
      class_number = CLASSES.get(label_from_file)
      if class_number is None:
        if label_from_file == OTHER:
          logging.info('Skipping {} label'.format(OTHER))
        else:
          raise Exception('Bad label {}'.format(label_from_file))
      else:
        bndbox = child.find('bndbox')
        xmin = (float(bndbox.find('xmin').text) / width)
        ymin = (float(bndbox.find('ymin').text) / height)
        xmax = (float(bndbox.find('xmax').text) / width)
        ymax = (float(bndbox.find('ymax').text) / height)

        ground_truth = SimpleNamespace(
          xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
          class_number=class_number, detected=False)
        ground_truths.append(ground_truth)

  return ground_truths


class Metrics:
  def __init__(self):
    self.labels = [[], [], [], []]
    self.scores = [[], [], [], []]
    self.box_in_wrong_place = 0
    self.total_predictions = 0

  def check_box(
      self, ground_truths, detection_score, detection_box, detection_class):
    found_matching_box = False
    for ground_truth in ground_truths:
      # The order for these coordinates comes from:
      # https://github.com/tensorflow/models/blob/
      # 62ce5d2a4c39f8e3add4fae70cb0d19d195265c6/research/
      # object_detection/utils/visualization_utils.py#L721
      ground_truth_bounding_box = [
        ground_truth.ymin,
        ground_truth.xmin,
        ground_truth.ymax,
        ground_truth.xmax
      ]
                
      if jaccard(detection_box, ground_truth_bounding_box) > JACCARD_THRESHOLD:
        if found_matching_box:
          print('Found multiple matching boxes')

        found_matching_box = True
        
        label_value = (1 if (detection_class == ground_truth.class_number)
                       else 0)
        self.labels[detection_class-1].append(label_value)
        self.scores[detection_class-1].append(detection_score)

    if not found_matching_box:
      self.box_in_wrong_place += 1

  def write_inference_and_ground_truth(self, graph, test_files):
    with graph.as_default():
      with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

        print('num files', len(test_files))
        for filename in test_files:
          image_path = os.path.join(IMG_DIR, '{}.JPEG'.format(filename))

          image = Image.open(image_path)
          if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
          width, height = image.size

          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)

          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})

          detection_classes = output_dict['detection_classes'][0].astype(np.uint8)
          detection_boxes = output_dict['detection_boxes'][0]
          detection_scores = output_dict['detection_scores'][0]

          ground_truths = load_ground_truths(filename, width, height)
                
          for detection_score, detection_box, detection_class in (
              zip(detection_scores, detection_boxes, detection_classes)):
            if detection_score > MIN_SCORE_THRESH:
              self.check_box(
                ground_truths, detection_score, detection_box,
                detection_class)
              self.total_predictions += 1

  def print_info():
    print('total predictions:', self.total_predictions)
    print('boxes in wrong place:', self.box_in_wrong_place)
        
    total_precision = 0
    for score, label in zip(self.scores, self.labels):
      total_precision += average_precision_score(score, label)

    mAP = total_precision / len(self.scores)
    print('mAP:', mAP)

  def store_results():
      with open('dont_walk_labels.p', 'wb') as pickle_file:
        pickle.dump(self.labels[0], pickle_file)

      with open('dont_walk_scores.p', 'wb') as pickle_file:
        pickle.dump(self.scores[0], pickle_file)

      with open('walk_labels.p', 'wb') as pickle_file:
        pickle.dump(self.labels[1], pickle_file)

      with open('walk_scores.p', 'wb') as pickle_file:
        pickle.dump(self.scores[1], pickle_file)

      with open('countdown_labels.p', 'wb') as pickle_file:
        pickle.dump(self.labels[2], pickle_file)

      with open('countdown_scores.p', 'wb') as pickle_file:
        pickle.dump(self.scores[2], pickle_file)

      with open('off_labels.p', 'wb') as pickle_file:
        pickle.dump(self.labels[3], pickle_file)

      with open('off_scores.p', 'wb') as pickle_file:
        pickle.dump(self.scores[3], pickle_file)    


def main():
  logging.basicConfig(level=logging.INFO)
  
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(
        '/home/ubuntu/cross_safe_april_2019/frcnn/exported_graphs/'
        'frozen_inference_graph.pb', 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

      with open('/home/ubuntu/cross_safe_april_2019/tfrecord_output/test_files.p', 'rb') as f:
        test_files = pickle.load(f)
        metrics = Metrics()
        metrics.write_inference_and_ground_truth(detection_graph, test_files)
        metrics.print_info()
        metrics.store_results()


if __name__ == '__main__':
  main()
