# import caption_generator
from keras.callbacks import ModelCheckpoint

from model import create_model
from pipeline import train_generator, generate_config


def train(batch_size=128,
          epochs=100,
          data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
          weights_path=None):
    '''Method to train the image caption generator
    weights_path is the path to the .h5 file where weights from the previous
    run are saved (if available)'''

    config_dict = generate_config()
    config_dict['batch_size'] = batch_size
    train_data_generator = train_generator(config_dict=config_dict,
                                           data_dir=data_dir)

    model = create_model()

    if weights_path:
        model.load_weights(weights_path)

    file_name = 'weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath=file_name,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(
        generator=train_data_generator,
        steps_per_epoch=config_dict['total_number_of_examples'] / batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list)


if __name__ == '__main__':
    train(batch_size=128,
          epochs=100,
          data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
          weights_path=None)
