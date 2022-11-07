# Object detection and classification

Tran a model to detect cloud types from satellite images. The data is taken from the [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization) Kaggle competition.

We use a [Unet model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) to assign every image pixel to up to four different cloud types. The data preprocessing and training is done with tensorflow.

## Data collection

Set `DATA_DIR`, `TFBOARD_DIR`, `ARTIFACTS_DIR` variables in `src/settings.py`. Get the data from [here](https://www.kaggle.com/competitions/understanding_cloud_organization/data). It should be put inside the `DATA_DIR` directory, as shown below:
```
$DATA_DIR
└── raw/
    ├── train_images/
    ├── test_images/
    ├── train.csv
    └── sample_submission.csv
```


## Data preprocessing

We create `.tfrecords` files where each record contains bytestrings of compressed jpeg-images and of the masks which
were compressed by storing them as png images. The dimensions of the images and masks are also reduced by the
reduction factor `rf`.
```shell
# tr, va, te: train, validation, test split fractions
python src/main.py create-datasets --tr=0.8 --va=0.1 --te=0.1 --rf=4
```
The train, validation, test datasets are stored in `DATA_DIR`, as shown below:
```
$DATA_DIR
├── raw/
└── preprocessed_rf_4/
    ├── train/
    ├── validation/
    ├── test/
    └── submission/  # dataset evaluated by Kaggle
```


## Training

The json strings that you can provide overwrite the default arguments used by the model.
```shell
python src/main.py train \
  --ds_args='{"batch_size": 64}' \
  --callbacks_args='{"profile_batch": [15,20], "period": 10}' \
  --training_args='{"epochs": 120, "verbose": 1}' \
  --rf=4
```
The results are stored in `ARTIFACTS_DIR`.
