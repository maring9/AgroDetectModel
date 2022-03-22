from tensorflow.keras.callbacks import ReduceLROnPlateau


def get_reducelr_callback(monitor='val_loss',
                          factor=0.1,
                          patience=10,
                          mode='auto',
                          min_delta=0.0001,
                          cooldown=0,
                          min_lr=0):
    """
        Helper function to create a callback to reduce te learning rate when a
        metric has stopped improving

    Args:
        monitor (str, optional):        Quantity to be monitored.
                                        Defaults to 'val_loss'.
        factor (float, optional):       Factor by which to reduce learning
                                        rate. Defaults to 0.1.
        patience (int, optional):       Number of epochs with no improvement
                                        after which the learning rate will be
                                        reduced. Defaults to 10.
        mode (str, optional):           Whether to save the models based on
                                        maximization or minimization of the
                                        monitored metric.
                                        Defaults to 'auto'.
        min_delta (float, optional):    Threshold for measuring the new
                                        optimum. Defaults to 0.0001.
        cooldown (int, optional):       Number of epochs to wait before
                                        resuming normal operation.
                                        Defaults to 0.
        min_lr (int, optional):         Lower bound on the learning rate.
                                        Defaults to 0.

    Returns:
        keras.callbacks.ReduceLROnPlateau:  ReduceLROnPlateau object
    """

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=factor,
                                  patience=patience,
                                  mode=mode,
                                  min_delta=min_delta,
                                  cooldown=cooldown,
                                  min_lr=min_lr)

    return reduce_lr
