# import caption_generator
from keras.callbacks import ModelCheckpoint, TensorBoard

from model import create_model
from pipeline import generate_config, debug_generator


def train(batch_size=128,
          epochs=100,
          data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
          weights_path=None,
          mode="train"):
    '''Method to train the image caption generator
    weights_path is the path to the .h5 file where weights from the previous
    run are saved (if available)'''

    config_dict = generate_config(data_dir=data_dir,
                                  mode=mode)
    config_dict['batch_size'] = batch_size
    steps_per_epoch = config_dict["total_number_of_examples"] // batch_size

    print("steps_per_epoch = ", steps_per_epoch)
    train_data_generator = debug_generator(config_dict=config_dict,
                                           data_dir=data_dir)

    model = create_model(config_dict=config_dict)

    if weights_path:
        model.load_weights(weights_path)

    file_name = data_dir + "model/weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath=file_name,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    tensorboard = TensorBoard(log_dir='../logs',
                              histogram_freq=0,
                              batch_size=batch_size,
                              write_graph=True,
                              write_grads=True,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)

    callbacks_list = [checkpoint, tensorboard]
    model.fit_generator(
        generator=train_data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks_list)


if __name__ == '__main__':
    train(batch_size=8,
          epochs=100,
          data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
          weights_path=None,
          mode="train")
