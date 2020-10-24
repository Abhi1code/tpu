# FashionPedia Flask Api

FashionPedia flask api for cloth and attribute segmentation.

## Setup

> pip install requirements.txt

> Download this <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-49.tar.gz">model</a> file and extract it in the current folder (Folder name must be `fashionpedia-spinenet-49`).

> Run command `python app.py` to start the server.

> Enter `localhost:5000/predict?uri=test.jpg` to get predictions, where uri can be any image url or local image file.

> Enter `localhost:5000/output?uri=test.jpg` to get rendered html and npy file, where uri can be any image url or local image file.

> Open `output.html` file in the browser to check rendered image and `output.npy` contains numpy array of predictions.

## Input Image
<img src="" />

## Output Image
<img src="" />
