# Based on:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


import numpy as np
import tensorflow as tf
import argparse
import pickle
import time
import os
import logging

from distutils.version import StrictVersion
from PIL import Image
from lxml import etree
from types import SimpleNamespace
from sklearn.metrics import average_precision_score
from object_detection.utils import visualization_utils as vis_util

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
ERROR_IMAGE_OUTPUT_DIR = 'error_images'
LABEL_NAMES = ['Don\'t Walk', 'Walk', 'Countdown', 'Off']
GROUND_TRUTH_COLOR = 'green'
PREDICTION_COLOR = 'red'

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def calc_iou(box_a, box_b):
  # From https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection/
  # blob/e5f32413fb56279d5b6fb51a243ca06e5c567dce/data_prep.py#L9
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
    self.multiple_matching_boxes = 0

  def store_image_with_bboxes(
      self, image, filename, ground_truths, detection_scores,
      detection_boxes, detection_classes):
    for ground_truth in ground_truths:
      label_display = LABEL_NAMES[ground_truth.class_number-1]
      # Adding extra lines to display_str_list so that label_display
      # will not get covered up by the detected box
      vis_util.draw_bounding_box_on_image(
        image,
        ground_truth.ymin,
        ground_truth.xmin,
        ground_truth.ymax,
        ground_truth.xmax,
        color=GROUND_TRUTH_COLOR,
        display_str_list=[label_display, label_display, label_display])
      
    for detection_score, detection_box, detection_class in (
        zip(detection_scores, detection_boxes, detection_classes)):
      label_display = LABEL_NAMES[detection_class-1]
      if detection_score > MIN_SCORE_THRESH:
        vis_util.draw_bounding_box_on_image(
          image,
          detection_box[0],
          detection_box[1],
          detection_box[2],
          detection_box[3],
          color=PREDICTION_COLOR,
          display_str_list=[label_display])

    image.save(os.path.join(
      ERROR_IMAGE_OUTPUT_DIR, '{}.JPEG'.format(filename)))
    
  def check_box(
      self, ground_truths, detection_score, detection_box, detection_class):
    found_matching_box = False
    box_mismatched = False
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

      if calc_iou(detection_box, ground_truth_bounding_box) > JACCARD_THRESHOLD:
        if found_matching_box:
          print('Found multiple matching boxes')
          self.multiple_matching_boxes += 1

        found_matching_box = True
        current_match = detection_class == ground_truth.class_number
        if not current_match:
          box_mismatched = True

        label_value = 1 if current_match else 0
        self.labels[detection_class-1].append(label_value)
        self.scores[detection_class-1].append(detection_score)

    if not found_matching_box:
      self.box_in_wrong_place += 1

    return ((not box_mismatched) and found_matching_box)

  def classify_images(self, graph, test_files, store_mistake_images):
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
          logging.info('starting %s at %f', filename, time.time())
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

          found_mistake = False
          for detection_score, detection_box, detection_class in (
              zip(detection_scores, detection_boxes, detection_classes)):
            if detection_score > MIN_SCORE_THRESH:

              # We are making mistake its own variable so self.check_box still runs
              # even when found_mistake is already true
              prediction_correct = self.check_box(
                ground_truths, detection_score, detection_box,
                detection_class)
              
              found_mistake = found_mistake or (not prediction_correct)
              self.total_predictions += 1

          if found_mistake and store_mistake_images:
            # We have already run image through the classifier, so we can draw
            # bounding boxes on it without affecting results
            self.store_image_with_bboxes(
              image, filename, ground_truths, detection_scores, detection_boxes, detection_classes)

  def print_info(self):
    print('total predictions:', self.total_predictions)
    print('boxes in wrong place:', self.box_in_wrong_place)
    print('multiple matching boxes:', self.multiple_matching_boxes)

    total_precision = 0
    num_nonempty_classes = 0
    for score, label in zip(self.scores, self.labels):
      
      assert len(score) == len(label), (
        'Mismatch of scores and labels')
      
      if len(label) == 0:
        print('Empty label array. Will be excluded from mAP')
      else:
        num_nonempty_classes += 1
        total_precision += average_precision_score(label, score)

    mAP = total_precision / num_nonempty_classes
    print('mAP:', mAP)
    print('Accross', num_nonempty_classes, 'classes')

  def store_results(self):
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
  parser = argparse.ArgumentParser(
      description='Run images through cross safe object detector.')
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--store-results', action='store_true')
  parser.add_argument('--mistake-images', action='store_true')

  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(level=logging.INFO)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(
        '/home/ubuntu/cross_safe_april_2019/ssd/exported_graphs/'
        'frozen_inference_graph.pb', 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

      with open('/home/ubuntu/cross_safe_april_2019/tfrecord_output/test_files.p', 'rb') as f:
        test_files = pickle.load(f)
        metrics = Metrics()
        metrics.classify_images(detection_graph, test_files, args.mistake_images)
        metrics.print_info()

        if args.store_results:
          metrics.store_results()

if __name__ == '__main__':
  main()
