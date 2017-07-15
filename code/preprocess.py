import pickle

import numpy as np

from vgg16 import load_image, load_vgg16

counter = 0

START = "__START__ "
END = " __END__"


def encode_image(model, image,
                 data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    '''Method to encode the given image'''
    image_dir = data_dir + "images/"
    prediction = model.predict(
        load_image(image_dir + str(image))
    )
    return np.reshape(prediction, prediction.shape[1])


def read_captions(data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
                  mode="all"):
    '''Method to read the captions from the text file'''
    with open(data_dir + "annotations/Flickr8k.token.txt") as caption_file:
        image_caption_dict = {}
        captions = map(lambda x: x.strip(),
                       caption_file.read().split('\n'))
        for caption in captions:
            image_name = caption.split('#')[0].strip()
            caption_text = caption.split('\t')[1].strip()
            if image_name not in image_caption_dict:
                image_caption_dict[image_name] = []
            image_caption_dict[image_name].append(START + caption_text + END)
    if (mode == "all"):
        return image_caption_dict
    else:
        image_name_list = read_image_list(mode=mode,
                                          data_dir=data_dir)
        filtered_image_caption_list = {}
        for image_name in image_name_list:
            filtered_image_caption_list[image_name] = image_caption_dict[image_name]
        return filtered_image_caption_list


def read_image_list(mode="train",
                    data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    '''Method to read the list of images'''
    with open(data_dir + "Flickr_8k." + mode + "Images.txt", 'r') as images_file:
        images = list(map(lambda x: x.strip(),
                          images_file.read().split('\n')))
    return images


def prepare_image_dataset(data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
                          mode_list=["train", "test", "debug"]):
    image_encoding_model = load_vgg16()
    for mode in mode_list:
        images = read_image_list(mode=mode,
                                 data_dir=data_dir)
        image_encoding = {}
        for image in images:
            image_encoding[image] = encode_image(image_encoding_model,
                                                 image,
                                                 data_dir=data_dir)
            with open(data_dir + "model/" + mode + "_image_encoding.pkl", "wb") as image_encoding_file:
                pickle.dump(image_encoding, image_encoding_file)


if __name__ == "__main__":
    data_dir="/home/shagun/projects/Image-Caption-Generator/data/"
    image_caption_dict = read_captions(data_dir=data_dir)
    prepare_image_dataset(data_dir=data_dir,
                          mode_list=["debug"])
