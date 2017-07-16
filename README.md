# Image-Caption-Generator
A simple implementation of neural image caption generator

Please note that the code in this repo is for use in talks/workshops. There is a lot of room for improvement (in terms of both accuracy and efficiency) so that these aspects can be discussed during the sessions.

You may refer to [Tensorflow's im2text Model](https://github.com/tensorflow/models/tree/master/im2txt) for a stable and accurate implementation.

## Setup

### Create Directories

* Run `./scripts/mkdir.sh`

### Downloading Datasets

* Run `./scripts/download_images.sh`
* This downloads Flick8K dataset

### Downloading Models

* The VGG16 model would be downloaded automatically when the model is trained for the first time and would be cached on the disk.
* Alternatively, run `python3 vgg16.py`. It would download the VGG16 model, produce the embeddings for a test image and compare with a pre-computed embedding.

### Processing Images

* Update `data_dir` in `code/preprocess.py` and set `mode_list=["train", "test", "debug"]`
* Run `python3 preprocess.py`

## Train

* Run `python3 train.py`