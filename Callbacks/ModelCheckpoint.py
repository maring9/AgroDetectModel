from tensorflow.keras.callbacks import ModelCheckpoint


def get_modelcheckpoint_callback(filepath,
                                 monitor='val_loss',
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto'):
    """
        Helper function to create a callback that saves the model weights.

    Args:
        filepath (string):                  Path to save the model
        monitor (str, optional):            Metric to monitor during training.
                                            Defaults to 'val_loss'.
        save_best_only (bool, optional):    Whether to only save the best
                                            model. Defaults to False.
        save_weights_only (bool, optional): Whether to only save the model
                                            weights. Defaults to False.
        mode (str, optional):               Whether to save the models based on
                                            maximization or minimization of the
                                            monitored metric.
                                            Defaults to 'auto'.

    Returns:
        keras.callbacks.ModelCheckpoint: ModelCheckpoint object
    """

    chekcpoint_callback = ModelCheckpoint(filepath=filepath,
                                          monitor=monitor,
                                          save_best_only=save_best_only,
                                          save_weights_only=save_weights_only,
                                          mode=mode)

    return chekcpoint_callback
