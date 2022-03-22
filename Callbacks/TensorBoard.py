from consts import BASE_LOG_DIR
from tensorflow.keras.callbacks import TensorBoard


def get_tensorboard_callback(log_dir=BASE_LOG_DIR,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=False,
                             update_freq='epoch'):
    """
        Helper function to create a callback that enables visualizations

    Args:
        log_dir (string, optional):     Path of the directory where to save log
                                        files. Defaults to BASE_LOG_DIR.
        histogram_freq (int, optional): Frequency (in epochs) at which to
                                        compute activation and weight
                                        histogram. Defaults to 0.
        write_graph (bool, optional):   Whether to visualize the graph.
                                        Defaults to True.
        write_images (bool, optional):  Whether to write model weighs to
                                        visualize. Defaults to False.
        update_freq (str, optional):    Frequency at which to write metrics
                                        and losses. Defaults to 'epoch'.

    Returns:
        keras.callbacks.TensorBoard: TensorBoard object
    """

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=histogram_freq,
                                       write_graph=write_graph,
                                       write_images=write_images,
                                       update_freq=update_freq)

    return tensorboard_callback
