from tensorflow.keras.callbacks import EarlyStopping


def get_earlystopping_callback(monitor='val_loss',
                               patience=0,
                               mode='auto',
                               baseline=None,
                               best_weights=False):
    """
        Helper function to create a callback that stops training when a
        monitored metric has stopped improving

    Args:
        monitor (str, optional):        Quantity to be monitored.
                                        Defaults to 'val_loss'.
        patience (int, optional):       Number of epochs with no improvement
                                        after which training stopps.
                                        Defaults to 0.
        mode (str, optional):           Whether the goal is to minimize or
                                        maximize the monitored metric.
                                        Defaults to 'auto'.
        baseline (_type_, optional):    Baseline value for monitored metric,
                                        training will stop if there is no
                                        improvement over the baseline.
                                        Defaults to None.
        best_weights (bool, optional):  Whether to restore model weights from
                                        the epoch with the best monitored
                                        metric. Defaults to False.

    Returns:
        keras.callbacks.EarlyStopping: EarlyStopping object
    """

    earlystopping_callback = EarlyStopping(monitor=monitor,
                                           patience=patience,
                                           mode=mode,
                                           baseline=baseline,
                                           restore_best_weights=best_weights)
    return earlystopping_callback
