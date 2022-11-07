import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from collections import ChainMap

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def split_str(x: str) -> list[int]:
    if type(x) == str:
        return [int(el) for el in x.split(' ')]
    else:
        return []


def preprocess_labels_file(
        path: str,
        data_dir: str,
        n_el: int = None
):
    """ Preprocess bboxes data """

    def getter(idx: int):
        """ Get idx from regex match """
        return lambda x: x.group(idx)

    pat = r'(\w+\.jpg)\_(\w+)'

    df = (
        pd.read_csv(path, nrows=4 * n_el if n_el else None)
        .assign(
            image=lambda x: x['Image_Label'].str.replace(pat, getter(1), regex=True),
            cloud_type=lambda x: x['Image_Label'].str.replace(pat, getter(2), regex=True),
            pixels=lambda x: (x[['cloud_type', 'EncodedPixels']]
                              .apply(lambda y: {y[0]: split_str(y[1])}, 1)))
        .groupby(['image'], as_index=False)
        .agg(pixels=('pixels', lambda x: dict(ChainMap(*x))))
        .assign(
            Sugar=lambda x: x['pixels'].map(lambda y: len(y['Sugar']) > 0),
            Gravel=lambda x: x['pixels'].map(lambda y: len(y['Gravel']) > 0),
            Flower=lambda x: x['pixels'].map(lambda y: len(y['Flower']) > 0),
            Fish=lambda x: x['pixels'].map(lambda y: len(y['Fish']) > 0))
        .sort_values(by=['image']))

    df['image_path'] = [os.path.join(data_dir, x) for x in df['image']]

    return df


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def mask_from_compact_notation(
        mask_compact: list[int],
        rows: int = 1400,
        columns: int = 2100
):
    """

    Parameters
    ----------
      mask_compact:
          the even elements correspond to pixel position of flattened 2d array

          the odd elements correspond to the interval length starting from the
          previous even element where the mask is 1;

          the flattened array is obtained by going through every col from left
          to right;
      rows: image height
      columns: image width
    """

    mask = np.zeros(shape=(rows * columns,), dtype=int)

    for pos, length in zip(mask_compact[::2], mask_compact[1::2]):
        p = pos - 1
        mask[p: p + length] = 1

    mask = mask.reshape(columns, rows)
    mask = mask.T

    return mask


def mask_from_compact_notation_inverse(mask: np.ndarray):
    """ Inverse fn of `mask_from_compact_notation()` """

    mask = (mask > 0) * 1  # map to 0 and 1
    mask = mask.T
    mask = mask.reshape(-1)

    mask_r_extent = np.concatenate((mask, [0]))  # always ends with 0
    mask_r_shift = np.concatenate(([0], mask))  # always starts with 0

    # positive only if the previous value is 0 and the current value is 1
    # negative only if the previous value is 1 and the current value is 0
    # the number of 1 and -1 elements is equal
    mask_01 = mask_r_extent - mask_r_shift

    start_idx = np.where(mask_01 == 1)[0]
    end_idx = np.where(mask_01 == -1)[0]

    assert len(start_idx) == len(end_idx)

    lengths_of_stripes_with_ones = end_idx - start_idx
    begins_of_stripes_with_ones = start_idx + 1

    compact_notation = list(zip(begins_of_stripes_with_ones,
                                lengths_of_stripes_with_ones))

    compact_notation = np.array(compact_notation).reshape(-1)

    return compact_notation.tolist()


def create_ex_proto(
        name: str,
        image_path: str,
        masks_compact: dict[str, list[int]],
        reduction_factor: int = 1
):
    """ Create a dataset element and map it to tf.train.Example

    To keep the disk-size of the tfrecords small we:
    - store the bytes of the compressed image
    - compress the masks by using tf.io.encode_png() that interprets them as
        greyscale images
    - optionally, we also resize both the images, and the masks
    """

    height = 1400
    width = 2100
    height_reduced = height // reduction_factor
    width_reduced = width // reduction_factor

    img_str = tf.io.read_file(image_path)  # Load image as a scalar string

    if reduction_factor > 1:
        img = tf.io.decode_jpeg(img_str)
        img = tf.image.resize(img, size=(height_reduced, width_reduced))
        img = tf.cast(tf.round(img), dtype=tf.uint8)
        img_str = tf.io.encode_jpeg(img)

    feat = {}

    # Iterate through 'Sugar', 'Gravel', 'Flower', 'Fish' masks
    for k, v in masks_compact.items():
        mask = mask_from_compact_notation(v, rows=height, columns=width)
        mask = tf.constant(mask[..., np.newaxis], dtype=tf.uint8)

        if reduction_factor > 1:
            mask = tf.image.resize(mask, size=(height_reduced, width_reduced))
            mask = tf.cast(tf.round(mask), dtype=tf.uint8)

        mask = tf.io.encode_png(mask)
        feat[k] = _bytes_feature(mask)

    feat['image_raw'] = _bytes_feature(img_str)
    feat['name'] = _bytes_feature(name.encode('utf-8'))

    proto = tf.train.Example(features=tf.train.Features(feature=feat))

    return proto


def create_tfrecords(
        idx: pd.Index,
        df: pd.DataFrame,
        output_dir: str,
        partition_size: int = 400,
        reduction_factor: int = 1
):
    """ Create tfrecords. Each record holds `partition_size` examples. """

    os.makedirs(output_dir)

    chunks = len(idx) / partition_size
    chunks = int(np.ceil(chunks))

    idx_chunks = np.array_split(idx.to_numpy(), chunks)

    for i, idx_chunk in enumerate(idx_chunks):

        logger.info(f'\npartition {i + 1} of {chunks}')

        output_path = os.path.join(output_dir, f'{i:02d}.tfrecords')

        with tf.io.TFRecordWriter(output_path) as writer:

            for _, r in df.loc[idx_chunk].iterrows():
                ex_proto = create_ex_proto(
                    name=r['image'],
                    image_path=r['image_path'],
                    masks_compact=r['pixels'],
                    reduction_factor=reduction_factor)

                writer.write(ex_proto.SerializeToString())


def train_val_test_split(
        df: pd.DataFrame,
        tr_va_te_frac: tuple[float, float, float],
        stratify_cols: list[str],
        random_state: int = None
):
    """ Split data into training, validation and test 10% parts """

    frac_tr, frac_va, frac_te = tr_va_te_frac

    assert frac_tr + frac_va + frac_te == 1

    idx_tr, idx_rest = train_test_split(
        df.index,
        train_size=frac_tr,
        stratify=df.loc[:, stratify_cols],
        random_state=random_state)

    idx_va, idx_te = train_test_split(
        df.loc[idx_rest].index,
        train_size=frac_va / (frac_va + frac_te),
        stratify=df.loc[idx_rest, stratify_cols],
        random_state=random_state)

    del idx_rest

    return idx_tr, idx_va, idx_te


def create_tr_va_te_datasets(
        source_dir: str,
        output_dir: str,
        tr_va_te_frac: tuple[float, float, float],
        n_el: int = None,
        reduction_factor: int = 1
):
    """ Create training, validation, test and submission datasets from the raw
    data and store them as .tfrecords

    Parameters
    ----------
      source_dir:
      output_dir:
      tr_va_te_frac: train, validation and test fractions
      n_el: number of images to use for the creation of the three datasets
      reduction_factor: reduction factor of the image dimensions

      Expected content in `source_dir`:
        source_dir
        ├── train_images/
        ├── test_images/
        ├── train.csv
        └── sample_submission.csv

      Generated output in `output_dir`:
        `output_dir`
        ├── train/
        ├── validation/
        ├── test/
        └── submission/
    """

    df = preprocess_labels_file(
        path=os.path.join(source_dir, 'train.csv'),
        data_dir=os.path.join(source_dir, 'train_images'),
        n_el=n_el)

    # The sample masks in the submission (sb) dataset have no meaning, but we
    # will keep them for simplicity
    df_sb = preprocess_labels_file(
        path=os.path.join(source_dir, 'sample_submission.csv'),
        data_dir=os.path.join(source_dir, 'test_images'))

    idx_tr, idx_va, idx_te = train_val_test_split(
        df=df,
        tr_va_te_frac=tr_va_te_frac,
        stratify_cols=['Sugar', 'Gravel', 'Flower', 'Fish'],
        random_state=12)
    idx_sb = df_sb.index

    output_tr = os.path.join(output_dir, 'train')
    output_va = os.path.join(output_dir, 'validation')
    output_te = os.path.join(output_dir, 'test')
    output_sb = os.path.join(output_dir, 'submission')

    args = {'reduction_factor': reduction_factor}
    create_tfrecords(idx=idx_tr, df=df, output_dir=output_tr, **args)
    create_tfrecords(idx=idx_va, df=df, output_dir=output_va, **args)
    create_tfrecords(idx=idx_te, df=df, output_dir=output_te, **args)
    create_tfrecords(idx=idx_sb, df=df_sb, output_dir=output_sb, **args)
