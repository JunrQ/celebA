import sys
import multiprocessing

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
import pytorch_lightning as pl

from utils import *

_depth_model_map = {
  18 : models.resnet18,
  34 : models.resnet34,
  50 : models.resnet50,
  101 : models.resnet101
}

class CelebAModel(pl.LightningModule):

  def __init__(self, depth,
               criterion,
               batch_size,
               config,
               num_classes=40,
               pretrained=True):
    super(CelebAModel, self).__init__()
    resnet = _depth_model_map[depth](pretrained=pretrained)
    modules = list(resnet.children())[:-2]
    self.backbone = nn.Sequential(*modules)
    self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    backbone_out_features = 512 if depth < 50 else 2048
    self.fc = nn.Linear(backbone_out_features, num_classes, bias=True)

    self.num_classes = num_classes
    self.config = config
    self.optimizer_config = config['optimizer']
    self.scheduler_config = config['scheduler']
    def _re_cast_criterion(x, y, *args, **kwargs):
      return criterion(x.float(), y.float(), *args, **kwargs)
    self.criterion = _re_cast_criterion
    self.num_workers = min(8, multiprocessing.cpu_count() // 4)

  def forward(self, image):
    out = self.backbone(image)
    out = self.adaptive_pool(out)
    n, c, _, _ = out.shape
    out = out.view((n, c))
    out = self.fc(out)
    out = torch.sigmoid(out)
    return out

  def training_step(self, batch, batch_idx):
    image, target = batch
    y_hat = self(image)
    loss = self.criterion(y_hat, target)
    result = pl.TrainResult(loss)
    return pl.TrainResult(loss)

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    acc = FM.accuracy(y_hat, y)
    result = pl.EvalResult(checkpoint_on=loss)
    result.log_dict({'val_acc': acc, 'val_loss': loss})
    return result

  def test_step(self, batch, batch_idx):
    x, _ = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    acc = FM.accuracy(y_hat, y)
    result = pl.EvalResult(checkpoint_on=loss)

    # Write to output file

    # result = self.validation_step(batch, batch_idx)
    # result.rename_keys({'val_acc': 'test_acc', 'val_loss': 'test_loss'})
    return result

  def configure_optimizers(self):
    opt = get_optimizer(self.optimizer_config, self.parameters())
    sch = { 'scheduler' : get_scheduler(opt, self.scheduler_config),
            'interval' : 'step' }
    return [opt], [sch]

  def prepare_data(self):
    config = self.config
    self.train_dataset = get_dataset(name='train', **config['train_dataset'])
    self.val_dataset = get_dataset(name='valid', **config['valid_dataset'])
    self.test_dataset = get_dataset(name='test', **config['test_dataset'])

  def train_dataloader(self):
    return DataLoader(self.train_dataset, shuffle=True,
                      batch_size=self.config['batch_size'],
                      num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, shuffle=False,
                      batch_size=self.config['batch_size'],
                      num_workers=min(2, self.num_workers // 2))

  def val_dataloader(self):
    return DataLoader(self.val_dataset, shuffle=False,
                      batch_size=self.config['batch_size'],
                      num_workers=min(2, self.num_workers // 2))

