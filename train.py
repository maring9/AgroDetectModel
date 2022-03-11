import tensorflow as tf
from horovod.tensorflow import keras as hvd

from callbacks import (get_earlystopping_callback,
                       get_modelcheckpoint_callback, get_tensorboard_callback)
from consts import TRAIN_DIR, VAL_DIR
from models import (get_alexnet_architecture, get_efficientnetb7_architecture,
                    get_inceptionresnetv2_architecture,
                    get_inceptionv3_architecture, get_mobilenetv2_architecture,
                    get_resnet152v2_architecture, get_vgg16_architecture)

# Horovod initialization for distributed learning
hvd.init()


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print('limited: ', gpu)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Batch
batch_size = 32

# Train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR,
                                                       image_size=(150, 150),
                                                       batch_size=batch_size,
                                                       seed=42)
# Val dataset
val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR,
                                                     image_size=(150, 150),
                                                     batch_size=batch_size,
                                                     seed=42)


# Normalized train and val dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# Models
InceptionResnetV2 = get_inceptionresnetv2_architecture(weights=None)
AlexNet = get_alexnet_architecture()
EfficientNetB7 = get_efficientnetb7_architecture(weights=None)
MobileNetV2 = get_mobilenetv2_architecture(weights=None)
VGG16_ = get_vgg16_architecture(weights=None)
ResNet152v2 = get_resnet152v2_architecture(weights=None)
InceptionV3 = get_inceptionv3_architecture(weights=None)

# InceptionResnetV2 Trained
# AlexNet Trained
# MobileNetV2 Trained
# Using Adam lr 0.001
# ResNet152v2
# InceptionV3
#EfficientNetB7

# VGG untrained
models = [InceptionV3]

for model in models:
    optimizer = tf.optimizers.Adam(learning_rate=1e-3 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    horo = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

# avg_worker = hvd.callbacks.MetricAverageCallback()
# cm = ConfusionMatrix(val_ds)

    es = get_earlystopping_callback(patience=10, best_weights=False)
    tb = get_tensorboard_callback()

    callbacks_ = [horo, tb, es]

    if hvd.rank() == 0:
        name = "Checkpoints/{}".format(model.name)
        path = name + "/{epoch:02d}-{val_accuracy:.2f}"
        mc = get_modelcheckpoint_callback(path)
        callbacks_.append(mc)

    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks_)
