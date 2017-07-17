# import caption_generator
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from model import create_model
from pipeline import generate_config, get_tokenizer
from preprocess import load_vgg16, load_image


def make_caption_human_readable(caption, index_to_word):
    '''Method to convert the caption vector into human readable format'''
    return (' '.join(
        list(
            map(lambda x: index_to_word[x], caption)
        )
    ))


def gen_captions(config,
                 model,
                 image_embedding,
                 tokenizer,
                 num_captions,
                 index_to_word):
    '''Method to generate the captions given a model, embeddings of the input image
    and a beam size'''
    START = "__START__"
    captions = tokenizer.texts_to_sequences(texts=[START])
    scores = [0.0]
    while (len(captions[0]) < config["max_caption_length"]):
        new_captions = []
        new_scores = []
        for caption, score in zip(captions, scores):
            caption_so_far = pad_sequences(sequences=[caption],
                                           maxlen=config["max_caption_length"],
                                           padding="pre")
            next_word_scores = model.predict(
                [image_embedding, np.asarray(caption_so_far)]
            )[0]
            candidate_next_words = np.argsort(next_word_scores)[-num_captions:]
            for next_word in candidate_next_words:
                caption_so_far, caption_score_so_far = caption[:], score
                caption_so_far.append(next_word)
                new_captions.append(caption_so_far)
                new_score = score + next_word_scores[next_word]
                new_scores.append(new_score)
        captions = new_captions
        scores = new_scores
        captions_scores_list = list(
            zip(captions, scores))
        captions_scores_list.sort(key=lambda x: x[1])
        captions_scores_list = captions_scores_list[-num_captions:]
        captions, scores = zip(*captions_scores_list)
    for (caption, score) in captions_scores_list:
        print("Generated caption is: ",
              make_caption_human_readable(caption=caption,
                                          index_to_word=index_to_word))
        print("Score is: ", str(score))
        print("==============")
    return captions_scores_list


def predict(image_name,
            data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
            weights_path=None,
            mode="test"):
    '''Method to predict the caption for a given image.
    weights_path is the path to the .h5 file (model)'''

    image_path = data_dir + "images/" + image_name
    vgg_model = load_vgg16()
    vgg_embedding = vgg_model.predict(
        load_image(image_path)
    )
    image_embeddings = [vgg_embedding]

    config_dict = generate_config(data_dir=data_dir,
                                  mode=mode)
    print(config_dict)

    model = create_model(config_dict=config_dict,
                         compile_model=False)

    model.load_weights(data_dir + "model/" + weights_path)

    tokenizer = get_tokenizer(config_dict=config_dict,
                              data_dir=data_dir)

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    for image_embedding in image_embeddings:
        gen_captions(config=config_dict,
                     model=model,
                     image_embedding=image_embedding,
                     tokenizer=tokenizer,
                     num_captions=2,
                     index_to_word=index_to_word
                     )


if __name__ == '__main__':
    weights_path = "weights-00.hdf5"
    image_name = "101669240_b2d3e7f17b.jpg"
    data_dir = "/home/shagun/projects/Image-Caption-Generator/data/"
    predict(image_name=image_name,
            data_dir=data_dir,
            weights_path=weights_path,
            mode="debug")
