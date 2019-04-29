# Copyright 2019 Carnegie Mellon University
#
# Based on:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


import numpy as np
import tensorflow as tf
import pickle
import time
import heapq

from distutils.version import StrictVersion
from PIL import Image


# From https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#L632
MIN_SCORE_THRESH = 0.5
JACCARD_THRESHOLD = 0.5


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


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Load a (frozen) Tensorflow model into memory.

# In[10]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/home/ubuntu/Cross-Safe/models/research/new_splitted_data/exported_graphs/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Helper code

# In[12]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[13]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
with open('/home/ubuntu/Cross-Safe/models/research/new_splitted_data/holdout_files.p', 'rb') as f:
    filenames = pickle.load(f)
TEST_IMAGE_PATHS = [
    '/home/ubuntu/Cross-Safe/data/new_splitted_data/{}.JPEG'.format(filename)
    for filename in filenames
]

# In[14]:


def write_inference_and_ground_truth(graph):
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

      multiple_boxes = []
      wrong_class = []
      no_box = []

      correct_single_box = 0
      total = 0

      jaccard_total = 0
      heap = []

      dont_walk_labels = []
      dont_walk_scores = []

      walk_labels = []
      walk_scores = []

      total_predictions = 0

      for image_path in TEST_IMAGE_PATHS:
        total += 1

        image = Image.open(image_path)
        image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        with open('times.txt', 'a+') as timefile:
          # Run inference
          start = time.time()
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})
          end = time.time()
          timefile.write('{}\n'.format(end - start))

          # num_detections = int(output_dict['num_detections'][0])
          detection_classes = output_dict[
            'detection_classes'][0].astype(np.uint8)
          detection_boxes = output_dict['detection_boxes'][0]
          detection_scores = output_dict['detection_scores'][0]

          label_without_jpeg = image_path[:-5]
          with open('{}.txt'.format(label_without_jpeg)) as f:
            true_class = int(f.readline())
            ground_truth_bounding_box_file = [
              int(num)
              for num in f.readline().split()
            ]

            # From https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#L721
            ground_truth_bounding_box = [
              ground_truth_bounding_box_file[1] / 300,
              ground_truth_bounding_box_file[0] / 400,
              ground_truth_bounding_box_file[3] / 300,
              ground_truth_bounding_box_file[2] / 400,
            ]

            found_true_class = False
            for detection_score, detection_box, detection_class in (
                zip(detection_scores, detection_boxes, detection_classes)):
              if detection_score > 0.5:
                total_predictions += 1

                label_value = 0
                correct_box = calc_iou(detection_box, ground_truth_bounding_box) > JACCARD_THRESHOLD
                if (true_class == detection_class and correct_box):
                  found_true_class = True
                  label_value = 1
                else:
                  label_value = 0

                if detection_class == 1:
                  dont_walk_labels.append(label_value)
                  dont_walk_scores.append(detection_score)
                else:
                  walk_labels.append(label_value)
                  walk_scores.append(detection_score)

            if not found_true_class:
              if detection_class == 1:
                dont_walk_labels.append(1)
                dont_walk_scores.append(0)
              else:
                walk_labels.append(1)
                walk_scores.append(0)

            if detection_scores[1] > MIN_SCORE_THRESH:
              multiple_boxes.append(image_path)
            else:
              if detection_scores[0] <= MIN_SCORE_THRESH:
                no_box.append(image_path)
              elif true_class != detection_classes[0]:
                wrong_class.append(image_path)

              else:
                correct_single_box += 1

                inference_bounding_box = detection_boxes[0]

                single_jaccard = calc_iou(inference_bounding_box, ground_truth_bounding_box)
                heapq.heappush(heap, (single_jaccard, image_path))
                jaccard_total += single_jaccard

      ordered_by_jaccard = []
      while len(heap) > 0:
        popped = heapq.heappop(heap)[1]
        ordered_by_jaccard.append(popped)
      print('ordered by jaccard', ordered_by_jaccard)

      print('multiple boxes', len(multiple_boxes), multiple_boxes)
      print('wrong class', len(wrong_class), wrong_class)
      print('no box', len(no_box), no_box)

      print('total', total)
      print('correct single box', correct_single_box)
      print('Average Jaccard for correct single box', jaccard_total / correct_single_box)

      print('total predictions', total_predictions)

      with open('dont_walk_labels.p', 'wb') as pickle_file:
        pickle.dump(dont_walk_labels, pickle_file)

      with open('dont_walk_scores.p', 'wb') as pickle_file:
        pickle.dump(dont_walk_scores, pickle_file)

      with open('walk_labels.p', 'wb') as pickle_file:
        pickle.dump(walk_labels, pickle_file)

      with open('walk_scores.p', 'wb') as pickle_file:
        pickle.dump(walk_scores, pickle_file)

      from sklearn.metrics import average_precision_score
      walk_average_precision = average_precision_score(walk_labels, walk_scores)
      dont_walk_average_precision = average_precision_score(dont_walk_labels, dont_walk_scores)

      total_precision = walk_average_precision + dont_walk_average_precision
      mAP = total_precision / 2

      print('mAP', mAP)


write_inference_and_ground_truth(detection_graph)
