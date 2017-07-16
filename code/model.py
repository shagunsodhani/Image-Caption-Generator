from keras.layers import LSTM, Activation, Embedding, Dense, \
    RepeatVector, TimeDistributed, Input, Merge
from keras.layers import concatenate
from keras.models import Sequential, Model


def _create_image_model(config_dict,
                        image_inputs):
    # image_model = Sequential()
    # image_model.add(Dense(units=config_dict['embedding_dim'],
    #                       input_dim=4096,
    #                       activation='relu'))
    #
    # image_model.add(RepeatVector(n=config_dict['max_caption_length']))
    # return image_model

    x = Dense(units=config_dict['embedding_dim'],
              input_dim=4096,
              activation='relu',
              name="image_model_fc")(image_inputs)
    outputs = RepeatVector(n=config_dict['max_caption_length'],
                           name="image_model_repeatvector")(x)
    return outputs


def _create_language_model(config_dict,
                           language_inputs):
    # language_model = Sequential()
    # language_model.add(Embedding(input_dim=config_dict['vocabulary_size'],
    #                              output_dim=256,
    #                              input_length=config_dict['max_caption_length']))
    # language_model.add(LSTM(256, return_sequences=True))
    # language_model.add(TimeDistributed(Dense(units=config_dict['embedding_dim'])))
    # return language_model

    x = Embedding(input_dim=config_dict['vocabulary_size'],
                  output_dim=256,
                  input_length=config_dict['max_caption_length'],
                  name="language_model_embedding")(language_inputs)

    x = LSTM(256,
             return_sequences=True,
             name="language_model_lstm")(x)

    outputs = TimeDistributed(Dense(units=config_dict['embedding_dim']),
                              name="language_model_td")(x)
    return outputs


def _create_merged_model(config_dict,
                         image_model,
                         language_model):
    print(image_model.summary())
    print(language_model.summary())
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat'))
    model.add(LSTM(1000, return_sequences=False))
    model.add(Dense(units=config_dict['vocabulary_size']))
    model.add(Activation('softmax'))

    print(model.summary())
    return model


def create_model(config_dict,
                 compile_model=True):
    image_inputs = Input(shape=(4096,), name="image_model_input")
    image_model = _create_image_model(config_dict=config_dict,
                                      image_inputs=image_inputs)

    language_inputs = Input(shape=(config_dict['max_caption_length'],),
                            name="language_model_input")
    language_model = _create_language_model(config_dict=config_dict,
                                            language_inputs=language_inputs)

    merged_input = concatenate([image_model, language_model],
                               name="concatenate_image_language")
    merged_input = LSTM(1000,
                        return_sequences=False,
                        name="merged_model_lstm")(merged_input)
    softmax_output = Dense(units=config_dict["vocabulary_size"],
                           activation="softmax",
                           name="merged_model_softmax")(merged_input)
    model = Model(inputs=[image_inputs,
                          language_inputs], outputs=softmax_output)
    print(model.summary())
    if (compile_model == True):
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
