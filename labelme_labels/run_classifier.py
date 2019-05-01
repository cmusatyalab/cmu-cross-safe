# Based on:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf
import argparse
import pickle
import time
import os
import logging
import heapq

from distutils.version import StrictVersion
from PIL import Image
from lxml import etree
from types import SimpleNamespace
from sklearn.metrics import average_precision_score
from object_detection.utils import visualization_utils as vis_util

# from https://github.com/agschwender/pilbox/issues/34#issuecomment-84093912
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda x: None

FROZEN_INFERENCE_GRAPH = (
  '/home/ubuntu/cross_safe_april_2019/frcnn/exported_graphs/'
  'frozen_inference_graph.pb')
TEST_FILES = '/home/ubuntu/cross_safe_april_2019/tfrecord_output/test_files.p'
TIMES_FILE = 'times.txt'
IMG_DIR = '/home/ubuntu/cross_safe_april_2019/images'
LABEL_DIR = '/home/ubuntu/cross_safe_april_2019/labels'
ERROR_IMAGE_OUTPUT_DIR = 'error_images'

# From https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#L632
MIN_SCORE_THRESH = 0.5
JACCARD_THRESHOLD = 0.5

CLASSES = {
  'Don\'t walk' : 1,
  'dont walk' : 1,
  'walk' : 2,
  'Walk' : 2,
  'countdown' : 3,
  'off' : 4,
}
OTHER = 'other'
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


def construct_tensor_dict():
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

  return tensor_dict


class Metrics:
  def __init__(self):
    self.labels = [[], [], [], []]
    self.scores = [[], [], [], []]
    self.predictions_above_threshold = 0
    self.images_with_mistakes = 0

    # Ground truth boxess for the current image
    # Gets reset every time self.load_ground_truths is called
    self.ground_truths = []

    self.times = []

  def run_detection(self, sess, tensor_dict, image):

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    start = time.time()
    output_dict = sess.run(
      tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    end = time.time()

    time_delta = end - start
    self.times.append(time_delta)

    detection_classes = output_dict['detection_classes'][0].astype(np.uint8)
    detection_boxes = output_dict['detection_boxes'][0]
    detection_scores = output_dict['detection_scores'][0]

    detections = []
    detections_heap = []

    # Heap Tie breaking solution suggested here:
    # https://stackoverflow.com/a/39504878/859277
    for tiebreak, (detection_class, box, score) in enumerate(
        zip(detection_classes, detection_boxes, detection_scores)):
      detection = SimpleNamespace(
        class_number=detection_class, box=box, score=score)
      detections.append(detection)
      heapq.heappush(detections_heap, (-score, tiebreak, detection))

    return (detections, detections_heap)

  def load_ground_truths(self, filename, width, height):
    self.ground_truths = []
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
          self.ground_truths.append(ground_truth)

  def store_image_with_bboxes(self, image, filename, detections):
    for ground_truth in self.ground_truths:
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

    for detection in detections:
      label_display = LABEL_NAMES[detection.class_number-1]
      if detection.score > MIN_SCORE_THRESH:
        vis_util.draw_bounding_box_on_image(
          image,
          detection.box[0],
          detection.box[1],
          detection.box[2],
          detection.box[3],
          color=PREDICTION_COLOR,
          display_str_list=[label_display])

    image.save(os.path.join(
      ERROR_IMAGE_OUTPUT_DIR, '{}.JPEG'.format(filename)))

  def check_box(self, detection):
    """
    Check to see if detection was correct

    Updates self.ground_truths, self.labels, and self.scores
    Returns true if check was successful, false otherwise
    """

    max_iou_for_matching = 0
    matching_ground_truth_with_max_iou = None
    for ground_truth in self.ground_truths:
      if (ground_truth.detected == False and
          ground_truth.class_number == detection.class_number):

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

        current_iou = calc_iou(detection.box, ground_truth_bounding_box)
        if current_iou > max_iou_for_matching:
          max_iou_for_matching = current_iou
          matching_ground_truth_with_max_iou = ground_truth

    self.scores[detection.class_number-1].append(detection.score)
    if max_iou_for_matching > JACCARD_THRESHOLD:
      # Found a box with enough overlap
      matching_ground_truth_with_max_iou.detected = True
      self.labels[detection.class_number-1].append(1)
      return True
    else:
      self.labels[detection.class_number-1].append(0)
      return False

  def classify_images(self, graph, test_files, store_mistake_images):
    with graph.as_default():
      with tf.Session() as sess:
        tensor_dict = construct_tensor_dict()

        print('num files', len(test_files))
        for filename in test_files:
          logging.info('starting %s at %f', filename, time.time())
          image_path = os.path.join(IMG_DIR, '{}.JPEG'.format(filename))

          image = Image.open(image_path)
          if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
          width, height = image.size

          self.load_ground_truths(filename, width, height)
          (detections, detections_heap) = self.run_detection(
            sess, tensor_dict, image)

          found_mistake = False
          while len(detections_heap) > 0:
            detection = heapq.heappop(detections_heap)[2]

            # We are a separate variable so self.check_box still runs
            # even when found_mistake is already true
            prediction_correct = self.check_box(detection)

            # We will only consider this a mistake if the score is high enough
            if detection.score > MIN_SCORE_THRESH:
              found_mistake = found_mistake or (not prediction_correct)
              self.predictions_above_threshold += 1

          for ground_truth in self.ground_truths:
            if not ground_truth.detected:
              self.labels[ground_truth.class_number-1].append(1)
              self.scores[ground_truth.class_number-1].append(0)
              found_mistake = True

          if found_mistake:
            self.images_with_mistakes += 1
            if store_mistake_images:
              # We have already run image through the classifier, so we can draw
              # bounding boxes on it without affecting results
              self.store_image_with_bboxes(image, filename, detections)

  def print_info(self):
    print('predictions above threshold:', self.predictions_above_threshold)
    print('images with mistakes:', self.images_with_mistakes)

    total_precision = 0
    num_nonempty_classes = 0
    for class_number, (label, score) in enumerate(
        zip(self.labels, self.scores)):

      assert len(label) == len(score), (
        'Mismatch of scores and labels')

      if len(label) == 0:
        print('Empty array for', LABEL_NAMES[class_number])
        print('Will be excluded from mAP')
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

  def write_times(self):
    with open(TIMES_FILE, 'a+') as timesfile:
      for time_delta in self.times:
        timesfile.write('{}\n'.format(time_delta))


def main():
  parser = argparse.ArgumentParser(
      description='Run images through cross safe object detector.')
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--store-results', action='store_true')
  parser.add_argument('--mistake-images', action='store_true')
  parser.add_argument('--write-times', action='store_true')

  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(level=logging.INFO)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

      with open(TEST_FILES, 'rb') as f:
        test_files = pickle.load(f)
        metrics = Metrics()
        metrics.classify_images(detection_graph, test_files, args.mistake_images)
        metrics.print_info()

        if args.store_results:
          metrics.store_results()
        if args.write_times:
          metrics.write_times()


if __name__ == '__main__':
  main()
