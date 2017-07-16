import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from preprocess import read_captions, read_image_list


def _get_captions_text(data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
                       mode="all"):
    '''Method to return a list of all the caption texts'''
    image_captions_dict = read_captions(data_dir=data_dir,
                                        mode=mode)
    captions = []
    for key_value in image_captions_dict.items():
        for caption in key_value[1]:
            captions.append(caption)
    return captions


def generate_config(data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
                    mode="all"):
    '''Method to generate the config dict based on caption data'''
    config_dict = {
        "embedding_dim": 128,
        "vocabulary_size": -1,
        "max_caption_length": -1,
        "batch_size": 100,
        "total_number_of_examples": -1
    }
    captions = _get_captions_text(data_dir=data_dir,
                                  mode="all")

    max_caption_length = max(list(
        map(lambda caption: len(caption.split(' ')), captions)
    ))
    config_dict["max_caption_length"] = max_caption_length
    token_set = set()
    for caption in captions:
        for word in caption.split(' '):
            token_set.add(word)
    vocabulary_size = len(token_set)
    config_dict["vocabulary_size"] = vocabulary_size
    print("Vocabulary size = ", vocabulary_size)
    print("Maximum Caption Length = ", max_caption_length)
    total_number_of_examples = 0
    captions = _get_captions_text(data_dir=data_dir,
                                  mode=mode)
    for caption in captions:
        total_number_of_examples += len(caption.split(' ')) - 1
    config_dict["total_number_of_examples"] = total_number_of_examples
    print("Total Number of Examples = ", total_number_of_examples)
    return config_dict


def train_generator(config_dict, data_dir):
    '''Method to prepare the training dataset for feeding into the model'''
    return data_generator(config_dict=config_dict,
                          data_dir=data_dir,
                          mode="train")


def test_generator(config_dict, data_dir):
    '''Method to prepare the testing dataset for feeding into the model'''
    return data_generator(config_dict=config_dict,
                          data_dir=data_dir,
                          mode="test")


def debug_generator(config_dict, data_dir):
    '''Method to prepare the overfitting dataset for feeding into the model'''
    return data_generator(config_dict=config_dict,
                          data_dir=data_dir,
                          mode="debug")


def get_tokenizer(config_dict, data_dir):
    '''Method to obtain the tokenizer'''
    tokenizer = Tokenizer(num_words=config_dict["vocabulary_size"])
    texts = _get_captions_text(data_dir=data_dir,
                               mode="all")
    tokenizer.fit_on_texts(texts=texts)
    return tokenizer


def data_generator(config_dict, data_dir, mode):
    '''Method to prepare the data for feeding into the model'''
    batch_size = config_dict["batch_size"]
    # tokenizer = Tokenizer(num_words=config_dict["vocabulary_size"])
    # texts = _get_captions_text(data_dir=data_dir,
    #                            mode=mode)
    # tokenizer.fit_on_texts(texts=texts)
    tokenizer = get_tokenizer(config_dict=config_dict,
                              data_dir=data_dir)

    image_list = read_image_list(mode=mode, data_dir=data_dir)
    image_encoding_dict = pickle.load(
        open(data_dir + "model/" + mode + "_image_encoding.pkl", "rb")
    )

    image_captions_dict = read_captions(data_dir=data_dir)
    images_encoding = []
    captions = []
    for (index, image_name) in enumerate(image_list):
        current_image_encoding = image_encoding_dict[image_name]
        for caption in image_captions_dict[image_name]:
            images_encoding.append(current_image_encoding)
            captions.append(caption)
    captions = tokenizer.texts_to_sequences(texts=captions)

    while True:
        counter = 0
        current_batch_image = []
        current_batch_captionsofar = []
        current_batch_nextword = []
        for (current_caption, current_image_encoding) in zip(captions, images_encoding):
            for (index, token) in enumerate(current_caption[:-1]):
                caption_seen_so_far = current_caption[:index + 1]
                # next_word = defualt_next_word
                next_word = np.zeros(tokenizer.num_words)
                next_word[current_caption[index + 1]] = 1
                current_batch_image.append(current_image_encoding)
                current_batch_captionsofar.append(caption_seen_so_far)
                current_batch_nextword.append(next_word)
                counter += 1
                if (counter == batch_size):
                    current_batch_captionsofar = pad_sequences(
                        sequences=current_batch_captionsofar,
                        maxlen=config_dict["max_caption_length"],
                        padding="pre"
                    )
                    yield (
                        [np.asarray(current_batch_image),
                         current_batch_captionsofar],
                        np.asarray(current_batch_nextword)
                    )
                    counter = 0
                    current_batch_image = []
                    current_batch_captionsofar = []
                    current_batch_nextword = []


if __name__ == "__main__":
    generate_config()
