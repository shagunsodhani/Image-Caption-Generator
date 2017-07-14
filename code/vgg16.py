import numpy as np
from tensorflow.contrib.keras.api.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.preprocessing import image


def load_vgg16():
    '''Method to load the VGG16 model'''
    base_model = VGG16(include_top=True,
                       weights='imagenet',
                       input_shape=(224, 224, 3))

    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(name="fc2").output)

    return model


def load_image(path):
    '''Method to load the image'''
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)


if __name__ == "__main__":
    model = load_vgg16()
    img_path = 'elephant.jpg'
    x = load_image(img_path)
    print(model.predict(x))
