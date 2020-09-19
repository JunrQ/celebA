import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class CelebAModel(Trainer):
  """TODO"""


def get_trainer(gpus, path, config, debug=False):
  checkpoint_callback = ModelCheckpoint(
    filepath=path,
    save_top_k=True,
    verbose=True,
    monitor='val_acc',
    mode='min',
    prefix='')
  early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min')
  return Trainer(checkpoint_callback=checkpoint_callback,
                 early_stop_callback=early_stop,
                 gpus=gpus,
                 default_root_dir=path,
                 fast_dev_run=debug,
                 terminate_on_nan=True,
                 **config)
