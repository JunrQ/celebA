import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_latest_ckpt, get_min_loss_ckpt

class CelebAModel(Trainer):
  """TODO"""


def get_trainer(gpus, path, config, resume_mode='latest', debug=False):
  checkpoint_callback = ModelCheckpoint(
    filepath="%s/{epoch}-{val_loss:.2f}" % path,
    save_top_k=True,
    save_last=True,
    verbose=True,
    monitor='val_acc',
    mode='max',
    prefix='')
  early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min')
  
  if resume_mode == 'latest':
    resume_func = get_latest_ckpt
  elif resume_mode == 'min_loss':
    resume_func = get_min_loss_ckpt
  else:
    raise ValueError("%s not supported" % resume_mode)
  return Trainer(checkpoint_callback=checkpoint_callback,
                 early_stop_callback=early_stop,
                 gpus=gpus,
                 resume_from_checkpoint=resume_func(path),
                 default_root_dir=path,
                 fast_dev_run=debug,
                 terminate_on_nan=True,
                 **config)
