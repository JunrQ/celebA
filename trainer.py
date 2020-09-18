from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class CelebAModel(Trainer):
  """TODO"""


def get_trainer(gpus, path, config):
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
  return Trainer(checkpoint_callback=[checkpoint_callback,
                                      early_stop],
                 default_root_path=os.getcwd(),
                 gpus=gpus,
                 resume_from_checkpoint=path,
                 **config)
