from numpy import asarray
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dense, Dropout,Flatten
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import os
import random
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
import pickle
from input_prep import get_all_input, get_input_from_file
import tensorflow as tf
from ourGenerator import OurGenerator
from tensorflow.keras import regularizers
from lossHistory import LossHistory


CHECKPOINT_PATH = "checkpoints/ours_3_10epochs"

print('is gpu available?')
print(tf.config.list_physical_devices('GPU'))


def get_model():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
    input_shape=(10, 240, 320, 3),padding ="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.7))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.7))
    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='images/baseline.png', show_shapes=True, show_layer_names=True)

    return model


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels)
    validation_generator = OurGenerator(partition['val'], labels)
    model = get_model()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/ours_3_10epochs_model')
    compressed_pickle('history/ours_3_10epochs.pickle', history.history)
    # plotLearningCurve(history)


if __name__ == "__main__":
    main()
