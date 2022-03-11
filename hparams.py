import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from consts import TRAIN_DIR, VAL_DIR

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'nadam']))

METRIC = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC, display_name='Accuracy')],
  )

  def train_val_model(hparams):
    model = ...

    batch_size = 32
    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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

    log_hparams = hp.KerasCallback(log_dir, hparams)

    model.fit = (x, y, ...)


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_val_model(hparams)
        tf.summary.scalar(METRIC, accuracy, step=1)


def main():
    session = 0

    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_OPTIMIZER: optimizer
        }

        run_name = "run -%d" % session

        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session += 1
