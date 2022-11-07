import os
import re
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import segmentation_models as sm
import skimage.transform as st

import settings
from callbacks import create_callbacks
from data.dataset import load_dataset
from data.preprocessing import mask_from_compact_notation_inverse

tfkc = tf.keras.callbacks
tfkl = tf.keras.layers

logger = logging.getLogger(__name__)


def gpu_memory_setup():
    """ Restrict the amount of GPU memory that can be allocated by TensorFlow"""

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=settings.GPU_MEMORY_LIMIT * 1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def get_best_checkpoint(
        checkpoint_dir: str,
        pattern=r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'
):
    """ Parse names of all checkpoints, extract the validation loss
    and return the checkpoint with the lowest loss
    """

    pattern = r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'

    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = map(lambda x: re.fullmatch(pattern, x), checkpoints)
    checkpoints = filter(lambda x: x is not None, checkpoints)
    best_checkpoint = min(checkpoints, key=lambda x: float(x.group(1)))

    checkpoint_name = best_checkpoint.group(0).removesuffix('.index')

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    return checkpoint_path


def get_experiment_id(logs_dir: str):
    """ Generate an unused experiment id by looking at the tensorboard entries """

    experiments = os.listdir(logs_dir)
    experiments = map(lambda x: re.fullmatch(r'ex_(\d{3})', x), experiments)
    experiments = filter(lambda x: x is not None, experiments)
    experiments = map(lambda x: int(x.group(1)), experiments)
    experiments = set(experiments)

    experiment_id = min(set(np.arange(1_000)) - experiments)

    logger.info(f'\n\nExperiment id: {experiment_id}\n\n')

    return experiment_id


def generate_submission(model, ds):
    """ Generate submission """

    names = tf.concat([n for n in ds.map(lambda img, mask, name: name)], axis=0)
    names = names.numpy()
    names = [name.decode() for name in names]

    # TODO: machine runs out of memory !!!
    # shape = (n, height, width, 4)
    masks = model.predict(ds.map(lambda img, mask, name: img))

    df = pd.DataFrame()
    df['Image_Label'] = [f'{n}_{c}' for n in names for c in settings.CLASSES]
    df['EncodedPixels'] = ''
    df.set_index(['Image_Label'], inplace=True)

    for name, mask in zip(names, masks):

        mask = st.resize(mask, (350, 525), preserve_range=True)

        for i, cloud in enumerate(settings.CLASSES):
            mask_compact = mask_from_compact_notation_inverse(mask[..., i])
            mask_compact = [str(x) for x in mask_compact]
            mask_compact = ' '.join(mask_compact)

            df.loc[f'{name}_{cloud}', 'EncodedPixels'] = mask_compact

    df = df.reset_index()
    df = df.sort_values(['Image_Label'])
    df = df.reset_index(drop=True)

    return df


def unet_model(height: int, width: int):
    """ Adapt a unet model to our problem

    https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
    """

    # input dimensions should be multiples of 32
    height_unet = int(np.ceil(height / 32)) * 32
    width_unet = int(np.ceil(width / 32)) * 32

    height_diff = height_unet - height
    width_diff = width_unet - width

    model_unet = sm.Unet(
        backbone_name='resnet18',  # 'resnet18'  'mobilenetv2'
        input_shape=(height_unet, width_unet, 3),
        encoder_weights='imagenet',
        encoder_freeze=True,
        classes=4,
        activation=None)

    x_input = tfkl.Input(shape=(height, width, 3), dtype=tf.float32)
    x = tfkl.ZeroPadding2D(padding=((0, height_diff), (0, width_diff)))(x_input)
    x = model_unet(x)
    x = tfkl.Cropping2D(cropping=((0, height_diff), (0, width_diff)))(x)

    model = tf.keras.Model(inputs=x_input, outputs=x)

    return model


def dice_coefficient(y_true, y_pred):
    """ Dice coefficient """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0., tf.float32)

    intersection = tf.math.reduce_sum(y_true * y_pred)
    cardinalities = tf.math.reduce_sum(y_true + y_pred)

    dice = 2. * intersection / cardinalities

    return dice


def train(
        ds_dir: str,
        ds_args: dict,
        callbacks_args: dict,
        training_args: dict
):
    """ Train and evaluate a model """

    gpu_memory_setup()

    os.makedirs(settings.TFBOARD_DIR, exist_ok=True)

    experiment_id = get_experiment_id(settings.TFBOARD_DIR)

    log_dir = os.path.join(settings.TFBOARD_DIR, f'ex_{experiment_id:03d}')
    save_dir = os.path.join(settings.ARTIFACTS_DIR, f'ex_{experiment_id:03d}')

    all_args = dict(
        dataset_args=ds_args,
        callbacks_args=callbacks_args,
        training_args=training_args)

    # ds_sb: submission dataset
    dir_tr = os.path.join(ds_dir, 'train')
    dir_va = os.path.join(ds_dir, 'validation')
    dir_te = os.path.join(ds_dir, 'test')
    dir_sb = os.path.join(ds_dir, 'submission')

    ds_tr = load_dataset(dir_tr, **{**ds_args, 'augmentation': True})
    ds_va = load_dataset(dir_va, **ds_args)
    ds_te = load_dataset(dir_te, **ds_args)
    ds_sb = load_dataset(dir_sb, **{**ds_args, 'keep_name': True})

    # Train and evaluate a model
    model = unet_model(height=350, width=525)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=True),
        metrics=[dice_coefficient])

    callbacks = create_callbacks(
        log_dir, save_dir, experiment_data=all_args, ds=ds_va, **callbacks_args)

    model.fit(ds_tr, validation_data=ds_va, callbacks=callbacks, **training_args)

    # Save the model with the best weights
    checkpoint_path = get_best_checkpoint(os.path.join(save_dir, 'checkpoints'))
    model.load_weights(checkpoint_path)
    model.save(save_dir)

    evaluate_va = model.evaluate(ds_va, return_dict=True)
    evaluate_te = model.evaluate(ds_te, return_dict=True)
    logger.info('Evaluation: \n'
                f'validation: {evaluate_va}\n'
                f'test:       {evaluate_te}')

    # Generate a Kaggle submission file
    df = generate_submission(model, ds_sb)
    df.to_csv(os.path.join(save_dir, 'submission.csv'))
