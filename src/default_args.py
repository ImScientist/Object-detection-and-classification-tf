ds_args = dict(
    batch_size=32,
    prefetch_size=-1,
    cycle_length=2,  # number of concurrently opened files
    max_files=None,  # max number of `.tfrecord` files to read from
    take_size=-1  # max number of elements
)

training_args = dict(
    epochs=50,
    verbose=0)

callbacks_args = dict(
    histogram_freq=0,
    reduce_lr_patience=20,
    early_stopping_patience=40,
    profile_batch=(10, 15),  # batches to profile; set to 0 to disable profiling
    verbose=0,
    period=5  # store model weights every `period` epoch
)
