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


def read_captions(data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    '''Method to read the captions from the text file'''
    with open(data_dir + 'annotations/Flickr8k.token.txt') as caption_file:
        image_caption_dict = {}
        captions = map(lambda x: x.strip(),
                       caption_file.read().split('\n'))
        for caption in captions:
            image_name = caption.split("#")[0].strip()
            caption_text = caption.split("\t")[1].strip()
            if image_name not in image_caption_dict:
                image_caption_dict[image_name] = []
            image_caption_dict[image_name].append(START + caption_text + END)
    return image_caption_dict


def read_image_list(mode="train",
                    data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    '''Method to read the list of images'''
    if (mode == "train"):
        with open(data_dir + 'Flickr_8k.trainImages.txt', 'r') as train_images_file:
            train_images = list(map(lambda x: x.strip(),
                                    train_images_file.read().split('\n')))
        return train_images
    else:
        with open(data_dir + 'Flickr_8k.testImages.txt', 'r') as test_images_file:
            test_images = list(map(lambda x: x.strip(),
                                   test_images_file.read().split('\n')))
        return test_images


def prepare_image_dataset(data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    train_images = read_image_list(mode="train")
    test_images = read_image_list(mode="test")

    image_encoding_model = load_vgg16()

    image_encoding = {}
    for image in train_images:
        image_encoding[image] = encode_image(image_encoding_model, image)

    with open(data_dir + "model/train_image_encoding.pkl", "wb") as image_encoding_file:
        pickle.dump(image_encoding, image_encoding_file)

    image_encoding = {}
    for image in test_images:
        image_encoding[image] = encode_image(image_encoding_model, image)

    with open(data_dir + "model/test_image_encoding.pkl", "wb") as image_encoding_file:
        pickle.dump(image_encoding, image_encoding_file)


if __name__ == "__main__":
    image_caption_dict = read_captions(data_dir="/home/shagun/projects/Image-Caption-Generator/data/")
