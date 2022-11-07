"""
    For more help use: `python src/main.py --help`
"""

import os
import json
import click
import logging

import settings
import default_args
from data.preprocessing import create_tr_va_te_datasets
from train import train as train_fn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--tr', default=.8, type=float, help='training fraction')
@click.option('--va', default=.1, type=float, help='validation fraction')
@click.option('--te', default=.1, type=float, help='test fraction')
@click.option('--rf', default=1, type=int, help='reduction factor of image dimensions')
def create_datasets(tr, va, te, rf):
    """ Preprocess data and split it into a training, validation and test
    datasets """

    assert tr + va + te == 1

    create_tr_va_te_datasets(
        source_dir=os.path.join(settings.DATA_DIR, 'raw'),
        output_dir=os.path.join(settings.DATA_DIR, f'preprocessed_rf_{rf}'),
        tr_va_te_frac=(tr, va, te),
        reduction_factor=rf)


@cli.command()
@click.option("--ds_args", default='{}', type=str, help="json string of the dataset args")
@click.option("--callbacks_args", default='{}', type=str, help="json string of the callback args")
@click.option("--training_args", default='{}', type=str, help="json string of the training args")
@click.option('--rf', default=1, type=int, help='reduction factor of image dimensions')
def train(ds_args, callbacks_args, training_args, rf):
    """ Train a model """

    # Update the default arguments with the cli-arguments
    ds_args = {**default_args.ds_args, **json.loads(ds_args)}
    callbacks_args = {**default_args.callbacks_args, **json.loads(callbacks_args)}
    training_args = {**default_args.training_args, **json.loads(training_args)}

    logger.info(
        f'ds_args: {json.dumps(ds_args, indent=2)}\n'
        f'callbacks_args: {json.dumps(callbacks_args, indent=2)}\n'
        f'training_args: {json.dumps(training_args, indent=2)}\n')

    train_fn(
        ds_dir=os.path.join(settings.DATA_DIR, f'preprocessed_rf_{rf}'),
        ds_args=ds_args,
        callbacks_args=callbacks_args,
        training_args=training_args)


if __name__ == "__main__":
    cli()
