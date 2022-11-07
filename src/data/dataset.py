import os
import glob
import logging
import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_ex_proto_fn(ex_proto_serialized):
    """ Parse the input tf.train.Example proto """

    feature_description = {
        'Sugar': tf.io.FixedLenFeature([], tf.string),
        'Gravel': tf.io.FixedLenFeature([], tf.string),
        'Flower': tf.io.FixedLenFeature([], tf.string),
        'Fish': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string)}

    ex_proto = tf.io.parse_single_example(
        serialized=ex_proto_serialized,
        features=feature_description)

    mask = tf.concat([
        tf.io.decode_png(ex_proto['Sugar'], channels=1),
        tf.io.decode_png(ex_proto['Gravel'], channels=1),
        tf.io.decode_png(ex_proto['Flower'], channels=1),
        tf.io.decode_png(ex_proto['Fish'], channels=1),
    ], axis=-1)

    image = tf.io.decode_jpeg(ex_proto['image_raw'])
    name = ex_proto['name']

    return {'image': image, 'mask': mask, 'name': name}


def remove_black_pixels_from_masks(el):
    """ Remove the areas in the mask where the image pixels are black

    Both image and mask hold values of type uint8.
    """

    image = tf.cast(el['image'], tf.float32)
    mask = tf.cast(el['mask'], tf.float32)

    non_black_px = tf.clip_by_value(
        tf.reduce_sum(image, axis=-1, keepdims=True),
        tf.constant(0, tf.float32),
        tf.constant(1, tf.float32))

    mask = mask * non_black_px

    name = el['name']

    return {'image': image, 'mask': mask, 'name': name}


def load_dataset(
        data_dir: str,
        batch_size: int,
        prefetch_size: int,
        cycle_length: int = 2,
        max_files: int = None,
        take_size: int = -1,
        augmentation: bool = False,
        keep_name: bool = False
):
    """ Make dataset from a directory with .tfrecords files

    Parameters
    ----------
    data_dir:
    batch_size:
    prefetch_size:
    cycle_length: number of files to read concurrently
    max_files: take all files if max_files=None
    take_size: take all elements of the dataset if take_size=-1
    augmentation: augment the data
    keep_name: keep image_name in the dataset
    """

    files = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    files = sorted(files)[:max_files]

    args = {'num_parallel_calls': tf.data.AUTOTUNE}

    ds = (
        tf.data.Dataset
        .from_tensor_slices(files)
        .interleave(
            lambda f: tf.data.TFRecordDataset(f),
            block_length=batch_size,
            cycle_length=cycle_length,
            **args)
        .take(take_size)
        .map(parse_ex_proto_fn, **args)
        .map(remove_black_pixels_from_masks, **args)
        .map(lambda x: (x['image'] / tf.constant(255, tf.float32),
                        x['mask'],
                        x['name']), **args)
        .batch(batch_size)
        .prefetch(prefetch_size)
    )

    if augmentation:
        ds = ds.map(lambda img, mask, name: (*augmentation_fn(img, mask), name),
                    **args)

    if not keep_name:
        ds = ds.map(lambda img, mask, name: (img, mask), **args)

    return ds


def augmentation_fn(img, mask):
    """ Image and mask augmentation

    Both image and mask hold values of type tf.float32.
    """

    seed_args = {'minval': tf.int32.min, 'maxval': tf.int32.max, 'dtype': tf.int32}

    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_flip_left_right(img, seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed)

    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_flip_up_down(img, seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed)

    # Image transformations that do not require a modification of the mask
    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_brightness(img, .2, seed)
    img = tf.clip_by_value(img, 0, 1)

    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_contrast(img, .8, 1.2, seed)
    img = tf.clip_by_value(img, 0, 1)

    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_saturation(img, 0.8, 1.2, seed)
    img = tf.clip_by_value(img, 0, 1)

    seed = tf.random.uniform((2,), **seed_args)
    img = tf.image.stateless_random_hue(img, 0.2, seed)
    img = tf.clip_by_value(img, 0, 1)

    return img, mask
