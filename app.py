from flask import Flask, request, jsonify, json
import logging as logger
logger.basicConfig(level="DEBUG")
import inference as infer
sess, predictions, image_input = infer.initiate()
#app = Flask(__name__)
#import numpy as geek
import urllib.request
import re
import tensorflow.compat.v1 as tf
import csv
# if __name__ == '__main__':
#     logger.debug("starting the application")
#     from api import *
#     app.run(host="127.0.0.1", port=5000, debug=True, load_dotenv=True)
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

import time

start = time.time()

start_count = 0
total_count = 0

if start_count == 0 or total_count == 0:
  raise ValueError('Start count or total count is not set')

while start_count <= total_count:
  infer.saveOutputs(sess, predictions, image_input, 'images/' + str(start_count) + '.png', label_map_dict, start_count)
  print("Processing image: " + str(start_count) + ".png")
  start_count += 1

#infer.saveOutputs(sess, predictions, image_input, 'pred.jpg', label_map_dict)

end = time.time()
print(end - start)

#@app.route("/", methods=["GET"])
def hello():
    return "FashionPedia endpoint -> /predict";

#@app.route("/predict", methods=["GET"])
def predict():
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(regex, request.args.get('uri')) is not None:
        urllib.request.urlretrieve(request.args.get('uri'), 'pred.jpg')
        res = infer.runModel(sess, predictions, image_input, 'pred.jpg')
        return res
    res = infer.runModel(sess, predictions, image_input, request.args.get('uri'))
    #print(res)
    #request.args.get('image')
    return res;

#@app.route("/output", methods=["GET"])
def output():
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(regex, request.args.get('uri')) is not None:
        urllib.request.urlretrieve(request.args.get('uri'), 'pred.jpg')
        res = infer.saveOutputs(sess, predictions, image_input, 'pred.jpg')
        return res
    res = infer.saveOutputs(sess, predictions, image_input, request.args.get('uri'))
    #print(res)
    #request.args.get('image')
    return res;

#app.run(host='0.0.0.0')