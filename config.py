class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    row_size = 32
    column_size = 32
    channel_size = 3
    n_classes = 10

    dropout = 0.5
    batch_size = 64
    lr = 0.001
