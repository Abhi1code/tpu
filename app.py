from flask import Flask, request, jsonify, json
import logging as logger
logger.basicConfig(level="DEBUG")
import inference as infer
sess, predictions, image_input = infer.initiate()
app = Flask(__name__)
#import numpy as geek
import urllib.request
import re
import calci as cal
from PIL import Image

# if __name__ == '__main__':
#     logger.debug("starting the application")
#     from api import *
#     app.run(host="127.0.0.1", port=5000, debug=True, load_dotenv=True)

@app.route("/", methods=["GET"])
def hello():
    return "FashionPedia endpoint -> /predict";

@app.route("/predict", methods=["GET"])
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

@app.route("/output", methods=["GET"])
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

@app.route("/category", methods=["GET"])
def category():
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
        res = cal.generate_all_cat(res)
        return res
    res = infer.runModel(sess, predictions, image_input, request.args.get('uri'))
    res = cal.generate_all_cat(res)
    #print(res)
    #request.args.get('image')
    return res;

app.run(host='0.0.0.0')