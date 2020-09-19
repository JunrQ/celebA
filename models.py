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
               path,
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
    self.path = path
    self.save_result_filename = os.path.join(self.path, 'predictions.txt')
    if os.path.isfile(self.save_result_filename):
      os.remove(self.save_result_filename)
    self.save_result_file = open(self.save_result_filename, 'a')

  def forward(self, image):
    out = self.backbone(image)
    out = self.adaptive_pool(out)
    n, c, _, _ = out.shape
    out = out.view((n, c))
    out = self.fc(out)
    out = torch.sigmoid(out)
    return out

  def training_step(self, batch, batch_idx):
    image, target, _ = batch
    y_hat = self(image)
    loss = self.criterion(y_hat, target)
    acc = FM.accuracy((y_hat > 0.5).long(), target)
    result = pl.TrainResult(loss)
    result.log_dict({'acc' : acc, 'loss' : loss}, on_epoch=True)
    return result

  # def training_epoch_end(self, outputs):
  #   print("[Train] Acc: %.3f" % outputs['acc'].mean().cpu().item())
  #   return outputs

  def validation_step(self, batch, batch_idx):
    x, y, _ = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    acc = FM.accuracy((y_hat > 0.5).long(), y)
    result = pl.EvalResult(checkpoint_on=loss)
    result.log_dict({'val_acc': acc, 'val_loss': loss})
    return result

  def test_step(self, batch, batch_idx):
    # Some part copied frm validation_step
    x, y, filename = batch
    y_hat = self(x)

    # Save result
    for i in range(y.shape[0]):
      self.save_result_file.write("%s %s\n" % (
          filename[i],
          ' '.join([str(1 if x > 0.5 else -1) for x in list(y_hat[i, ...].cpu().numpy())])))

    loss = self.criterion(y_hat, y)
    acc = FM.accuracy((y_hat > 0.5).long(), y)
    result = pl.EvalResult(checkpoint_on=loss)
    result.log_dict({'test_acc': acc, 'test_loss': loss})
    return result

  def configure_optimizers(self):
    opt = get_optimizer(self.optimizer_config, self.parameters())
    sch = { 'scheduler' : get_scheduler(opt, self.scheduler_config),
            'interval' : 'epoch' }
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
                      num_workers=min(4, self.num_workers // 2))

  def val_dataloader(self):
    return DataLoader(self.val_dataset, shuffle=False,
                      batch_size=self.config['batch_size'],
                      num_workers=min(4, self.num_workers // 2))

