# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.

It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
import tensorflow.compat.v1 as tf

from dataloader import mode_keys
from configs import factory as config_factory
from modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'attribute_mask_rcnn', 'Support `attribute_mask_rcnn`.')
flags.DEFINE_integer('image_size', 640, 'The image size.')
flags.DEFINE_string(
    'checkpoint_path', 'fashionpedia-spinenet-49/model.ckpt', 'The path to the checkpoint file.')
flags.DEFINE_string(
    'config_file', 'configs/yaml/spinenet49_amrcnn.yaml', 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_string(
    'label_map_file', 'dataset/fashionpedia_label_map.csv',
    'The label map file. See --label_map_format for the definition.')
flags.DEFINE_string(
    'label_map_format', 'csv',
    'The format of the label map file. Currently only support `csv` where the '
    'format of each row is: `id:name`.')
flags.DEFINE_string(
    'image_file_pattern', '',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'output_html', '/tmp/test.html',
    'The output HTML file that includes images with rendered detections.')
flags.DEFINE_string(
    'output_file', '/tmp/res.npy',
    'The output npy file that includes model output.')
flags.DEFINE_integer(
    'max_boxes_to_draw', 20, 'The maximum number of boxes to draw.')
flags.DEFINE_float(
    'min_score_threshold', 0.05,
    'The minimum score thresholds in order to draw boxes.')



def initiate():
  # Load the label map.
  print(' - Loading the label map...')
  label_map_dict = {}
  if 'csv' == 'csv':
    with tf.gfile.Open('dataset/fashionpedia_label_map.csv', 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=':')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id:name` format.')
        id_index = int(row[0])
        name = row[1]
        label_map_dict[id_index] = {
            'id': id_index,
            'name': name,
        }
  else:
    raise ValueError(
        'Unsupported label map format: {}.'.format('csv'))

  params = config_factory.config_generator('attribute_mask_rcnn')
  if 'configs/yaml/spinenet49_amrcnn.yaml':
    params = params_dict.override_params_dict(
        params, 'configs/yaml/spinenet49_amrcnn.yaml', is_strict=True)
  params = params_dict.override_params_dict(
      params, '', is_strict=True)
  params.override({
      'architecture': {
          'use_bfloat16': False,  # The inference runs on CPU/GPU.
      },
  }, is_strict=True)
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)

  with tf.Graph().as_default():
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [640, 640]
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(
        images, {'image_info': images_info}, mode=mode_keys.PREDICT)

    outputs['detection_boxes'] = (
        outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()
    sess = tf.Session()
    print(' - Loading the checkpoint...')
    saver.restore(sess, 'fashionpedia-spinenet-49/model.ckpt')
    print(' - Checkpoint Loaded...')
    return sess, predictions, image_input

def runModel(sess, predictions, image_input, image_file):

    print(' - Processing image ...')
    with tf.gfile.GFile(image_file, 'rb') as f:
        image_bytes = f.read()

    image = Image.open(image_file)
    image = image.convert('RGB')  # needed for images with 4 channels.
    width, height = image.size
    np_image = (np.array(image.getdata())
                .reshape(height, width, 3).astype(np.uint8))

    predictions_np = sess.run(
        predictions, feed_dict={image_input: image_bytes})

    num_detections = int(predictions_np['num_detections'][0])
    np_boxes = predictions_np['detection_boxes'][0, :num_detections]
    np_scores = predictions_np['detection_scores'][0, :num_detections]
    np_classes = predictions_np['detection_classes'][0, :num_detections]
    np_classes = np_classes.astype(np.int32)
    np_attributes = predictions_np['detection_attributes'][
                    0, :num_detections, :]

    np_masks = None
    if 'detection_masks' in predictions_np:
        instance_masks = predictions_np['detection_masks'][0, :num_detections]
        np_masks = mask_utils.paste_instance_masks(
            instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width)
        encoded_masks = [
            mask_api.encode(np.asfortranarray(np_mask))
            for np_mask in list(np_masks)]

    res = {
        'image_file': image_file,
        'boxes': np_boxes.tolist(),
        'classes': np_classes.tolist(),
        'scores': np_scores.tolist(),
        'attributes': np_attributes.tolist(),
        'masks': encoded_masks
    }
    #'masks': encoded_masks,
    print("Output generated")
    return res

def saveOutputs(sess, predictions, image_input, image_file, label_map_dict, fname):
    

    print(' - Processing image ...')
    with tf.gfile.GFile(image_file, 'rb') as f:
        image_bytes = f.read()

    image = Image.open(image_file)
    image = image.convert('RGB')  # needed for images with 4 channels.
    width, height = image.size
    np_image = (np.array(image.getdata())
                .reshape(height, width, 3).astype(np.uint8))
    image1 = Image.new('RGB', (width, height))
    #image1.save("image.png", "PNG")
    np_image1 = (np.array(image1.getdata())
                .reshape(height, width, 3).astype(np.uint8))          

    predictions_np = sess.run(
        predictions, feed_dict={image_input: image_bytes})

    num_detections = int(predictions_np['num_detections'][0])
    np_boxes = predictions_np['detection_boxes'][0, :num_detections]
    np_scores = predictions_np['detection_scores'][0, :num_detections]
    np_classes = predictions_np['detection_classes'][0, :num_detections]
    np_classes = np_classes.astype(np.int32)
    np_attributes = predictions_np['detection_attributes'][
                    0, :num_detections, :]

    np_masks = None
    if 'detection_masks' in predictions_np:
        instance_masks = predictions_np['detection_masks'][0, :num_detections]
        np_masks = mask_utils.paste_instance_masks(
            instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width)
        encoded_masks = [
            mask_api.encode(np.asfortranarray(np_mask))
            for np_mask in list(np_masks)]

    res = []
    res.append({
        'image_file': image_file,
        'boxes': np_boxes,
        'classes': np_classes,
        'scores': np_scores,
        'attributes': np_attributes
    })
    #'masks': encoded_masks,
    print("Output generated")

    image_with_detections_list = []
    image_with_detections = (
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            np_image1,
            np_boxes,
            np_classes,
            np_scores,
            label_map_dict,
            instance_masks=np_masks,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=0.05))
    ii = Image.fromarray(image_with_detections)        
    ii.save('mesh/' + str(fname) + '.png', "PNG") 
    image_with_detections_list.append(image_with_detections)
    return {"v": "output.html", "d": "output.npy"}


#if __name__ == '__main__':
  #logging.set_verbosity(logging.INFO)
  #tf.app.run(initiate)
  #sess, predictions, image_input = initiate()
  #res = runModel(sess, predictions, image_input, 'test.jpg')


