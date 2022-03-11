import tensorflow as tf

from callbacks import (ConfusionMatrix, get_earlystopping_callback, get_modelcheckpoint_callback,
                       get_tensorboard_callback)
from consts import TRAIN_DIR, VAL_DIR
from models import get_inceptionresnetv2_architecture

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = datagen.flow_from_directory(TRAIN_DIR,
                                              target_size=(150, 150),
                                              batch_size=batch_size,
                                              class_mode='sparse',
                                              shuffle=True,
                                              seed=42)

val_generator = datagen.flow_from_directory(VAL_DIR,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='sparse',
                                            shuffle=True,
                                            seed=42)

model = get_inceptionresnetv2_architecture(weights=None)

tb = get_tensorboard_callback()
cm = ConfusionMatrix(val_generator)

mc = get_modelcheckpoint_callback('Checkpoints/InceptionResnetRun2/{epoch}',
                                  save_best_only=True,
                                  save_weights_only=True)
es = get_earlystopping_callback(patience=15, best_weights=True)

model.fit(train_generator, epochs=50, callbacks=[tb, cm, mc, es])
