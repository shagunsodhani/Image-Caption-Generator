from keras.layers import LSTM, Merge, Activation, Input, \
    Embedding, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential, Model


def _create_image_model(config_dict):
    # image_model = Sequential()
    # image_model.add(Dense(units=config_dict['embedding_dim'],
    #                       input_dim=4096,
    #                       activation='relu'))
    #
    # image_model.add(RepeatVector(n=config_dict['max_caption_length']))
    # return image_model

    inputs = Input(shape=(4096,), name="image_model_input")
    x = Dense(units=config_dict['embedding_dim'],
              input_dim=4096,
              activation='relu',
              name="image_model_fc")(inputs)
    outputs = RepeatVector(n=config_dict['max_caption_length'],
                           name="image_model_repeatvector")(x)
    return Model(inputs=inputs,
                 outputs=outputs,
                 name="image_model")


def _create_language_model(config_dict):
    # language_model = Sequential()
    # language_model.add(Embedding(input_dim=config_dict['vocabulary_size'],
    #                              output_dim=256,
    #                              input_length=config_dict['max_caption_length']))
    # language_model.add(LSTM(256, return_sequences=True))
    # language_model.add(TimeDistributed(Dense(units=config_dict['embedding_dim'])))
    # return language_model

    inputs = Input(shape=config_dict['max_caption_length'],
                   name="language_model_input")

    x = Embedding(input_dim=config_dict['vocabulary_size'],
                  output_dim=256,
                  input_length=config_dict['max_caption_length'],
                  name="language_model_embedding")(inputs)

    x = LSTM(256,
             return_sequences=True,
             name="language_model_lstm")(x)

    outputs = TimeDistributed(Dense(units=config_dict['embedding_dim']),
                              name="language_model_td")(x)

    return Model(inputs=inputs,
                 outputs=outputs,
                 name="language_model")


def _create_merged_model(config_dict,
                         image_model,
                         language_model):
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat'))
    model.add(LSTM(1000, return_sequences=False))
    model.add(Dense(units=config_dict['vocabulary_size']))
    model.add(Activation('softmax'))
    return model


def create_model(config_dict,
                 ret_model=False):
    image_model = _create_image_model(config_dict=config_dict)

    language_model = _create_language_model(config_dict=config_dict)

    merged_model = _create_merged_model(config_dict=config_dict,
                                        image_model=image_model,
                                        language_model=language_model)
    if (ret_model == True):
        return merged_model

    merged_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    return merged_model
