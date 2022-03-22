import tensorflow as tf
from horovod.tensorflow import keras as hvd

from callbacks import (get_earlystopping_callback,
                       get_modelcheckpoint_callback, get_tensorboard_callback)
from consts import BATCH_SIZE, EPOCHS, IMAGE_SIZE, TRAIN_DIR, VAL_DIR
from utils import MODELS

# Horovod initialization for distributed learning
hvd.init()

# GPU configuration for distributed learning
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print('limited: ', gpu)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR,
                                                       image_size=IMAGE_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       seed=42)
# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR,
                                                     image_size=IMAGE_SIZE,
                                                     batch_size=BATCH_SIZE,
                                                     seed=42)


# Normalization layer used to normalize train / val dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Normalizing train and validation dataset
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# Main loop that trains all the model architectures established
for architecture in list(MODELS.keys()):
    # Load model architecture
    model = MODELS[architecture]

    # Initialize optimizer used for model training
    # Learning rate is scaled by the number of horovod workers
    optimizer = tf.optimizers.Adam(learning_rate=1e-3 * hvd.size())

    # Stock tensorflow optimizer is wrapped as horovod optimizer
    # used for distributed learning
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Compile model with the optimizer, loss function and metric
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Horovod callback used to broadcast all the initial gradients
    # to the same starting point
    hvd_init = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

    # Callback to stop training if the model does not improve
    early_stopping = get_earlystopping_callback(patience=10,
                                                best_weights=False)

    # Tensorboard callback used for monitoring and vizualization during
    # model training
    tensorboard = get_tensorboard_callback()

    # List of callbacks
    callbacks_ = [hvd_init, tensorboard, early_stopping]

    # Only one hvd worker should save the checkpoints to avoid file corruption
    if hvd.rank() == 0:
        # Model directory where to save the checkpoints
        name = "Checkpoints/{}".format(model.name)

        # Full checkpoint path and name
        path = name + "/{epoch:02d}-{val_accuracy:.2f}"

        # Callback used to save model checkpoints during training
        model_checkpoint = get_modelcheckpoint_callback(path)

        callbacks_.append(model_checkpoint)

    # Train model
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=callbacks_)
