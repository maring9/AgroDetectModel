from gc import callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from horovod.tensorflow import keras as hvd
from tensorflow import keras

from callbacks import (ConfusionMatrix, Metrics, get_earlystopping_callback,
                       get_modelcheckpoint_callback, get_tensorboard_callback)
from consts import (IMAGE_DIMS, NUM_CLASSES, TEST_DIR, TEST_IMAGES, TEST_LABELS,
                    TRAIN_DIR, VAL_DIR)
from models import get_inceptionresnetv2_architecture

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR,
                                                    image_size=(256, 256),
                                                    batch_size=batch_size,
                                                    seed=42)

val_ds = keras.utils.image_dataset_from_directory(VAL_DIR,
                                                  image_size=(256, 256),
                                                  batch_size=batch_size,
                                                  seed=42)
"""
test_ds = keras.utils.image_dataset_from_directory(TEST_DIR,
                                                   image_size=(256, 256),
                                                   batch_size=batch_size,
                                                   seed=42)
"""
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
net = get_inceptionresnetv2_architecture(weights=None)

optimizer = tf.optimizers.Adam(0.001 * hvd.size())

optimizer = hvd.DistributedOptimizer(optimizer)

net.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

horo = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

avg_worker = hvd.callbacks.MetricAverageCallback()
es = get_earlystopping_callback(patience=15, best_weights=True)
tb = get_tensorboard_callback()

# test_images = np.load(TEST_IMAGES)
# test_labels = np.load(TEST_LABELS)

cm = ConfusionMatrix(val_ds)
# metrics = Metrics((test_images, test_labels))

callbacks_ = [horo, avg_worker, tb, es, cm]

if hvd.rank() == 0:
    mc = get_modelcheckpoint_callback('Checkpoints/InceptionResnet/{epoch}.h5',
                                      save_best_only=True,
                                      save_weights_only=True)
    callbacks_.append(mc)

net.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=callbacks_)
