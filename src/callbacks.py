import os
import io
import json
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from visualization import visualize_multiple_predictions

tfkc = tf.keras.callbacks


def pretty_json(hp):
    """ Map dict to a json string """

    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def log_experiment_parameters(log_data: dict, file_writer):
    """ Log experiment parameters """

    with file_writer.as_default():
        log_data = pretty_json(log_data)
        tf.summary.text("experiment_args", log_data, step=0)


def log_model_architecture(model, file_writer):
    """ Store a non-interactive readable model architecture """

    with tempfile.NamedTemporaryFile('w', suffix=".png") as temp:
        _ = tf.keras.utils.plot_model(
            model,
            to_file=temp.name,
            show_shapes=True,
            dpi=64)

        im_frame = Image.open(temp.name)
        im_frame = np.asarray(im_frame)

        """ Log the figure """

        with file_writer.as_default():
            tf.summary.image(
                "model summary",
                tf.constant(im_frame, dtype=tf.uint8)[tf.newaxis, ...],
                step=0)


def plot_to_image(figure):
    """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this
        call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


class DisplayCallback(tf.keras.callbacks.Callback):
    """ Store the predicted segments on a predefined set of images

    Store the model architecture and experiment parameters in the beginning of
    the training
    """

    def __init__(self, log_dir, ds, n, experiment_data, period):
        super(DisplayCallback, self).__init__()
        self.ds = ds.unbatch().batch(1).take(8).cache()
        self.n = n
        self.period = max(period, 1)
        save_dir = os.path.join(log_dir, 'train')
        self.file_writer = tf.summary.create_file_writer(save_dir)
        self.experiment_data = experiment_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        y_hat = self.model.predict(self.ds.map(lambda x, y: x))
        y_ = tf.concat([y for y in self.ds.map(lambda x, y: y)], axis=0)

        fig = visualize_multiple_predictions(y_, (y_hat > 0) * 1, n=self.n)

        img = plot_to_image(fig)

        with self.file_writer.as_default():
            tf.summary.image('sample predictions', img, step=epoch)

    def on_train_begin(self, logs=None):
        log_model_architecture(
            model=self.model,
            file_writer=self.file_writer)

        log_experiment_parameters(
            log_data=self.experiment_data or {},
            file_writer=self.file_writer)


def create_callbacks(
        log_dir: str,
        save_dir: str = None,
        ds=None,
        histogram_freq: int = 0,
        reduce_lr_patience: int = 100,
        profile_batch: tuple = (10, 15),
        verbose: int = 0,
        early_stopping_patience: int = 250,
        period: int = 10,
        experiment_data: dict = None
):
    """ Generate model training callbacks """

    callbacks = [
        tfkc.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            profile_batch=profile_batch)]

    if ds is not None:
        callbacks.append(DisplayCallback(
            log_dir=log_dir, ds=ds, n=5, experiment_data=experiment_data, period=period))

    if reduce_lr_patience is not None:
        callbacks.append(
            tfkc.ReduceLROnPlateau(
                factor=0.2,
                patience=reduce_lr_patience,
                verbose=verbose))

    if early_stopping_patience is not None:
        callbacks.append(
            tfkc.EarlyStopping(patience=early_stopping_patience))

    if save_dir:
        path = os.path.join(
            save_dir,
            'checkpoints',
            'epoch_{epoch:03d}_loss_{val_loss:.4f}_cp.ckpt')

        callbacks.append(
            tfkc.ModelCheckpoint(
                path,
                save_weights_only=True,
                save_best_only=False,
                period=period))

    return callbacks
