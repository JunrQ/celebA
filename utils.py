import os
import PIL
import re
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class CustomCelebA(CelebA):
  """
  Rewrite `__gettiem__` since we need filename.
  """
  def __getitem__(self, index: int):
    X = PIL.Image.open(os.path.join(self.root, self.base_folder,
                                    "img_align_celeba", self.filename[index]))
    target: Any = []
    for t in self.target_type:
      if t == "attr":
        target.append(self.attr[index, :])
      elif t == "identity":
        target.append(self.identity[index, 0])
      elif t == "bbox":
        target.append(self.bbox[index, :])
      elif t == "landmarks":
        target.append(self.landmarks_align[index, :])
      else:
        # TODO: refactor with utils.verify_str_arg
        raise ValueError("Target type \"{}\" is not recognized.".format(t))

    if self.transform is not None:
      X = self.transform(X)

    if target:
      target = tuple(target) if len(target) > 1 else target[0]

      if self.target_transform is not None:
        target = self.target_transform(target)
    else:
      target = None
    return X, target, self.filename[index]


def get_dataset(path, transform, name='train', target_type='attr'):
  transform = parse_transform(transform)
  return CustomCelebA(root=path, split=name, target_type=target_type,
                      transform=transform, download=True)


def get_optimizer(config, parameters):
  config = config.copy()
  t = config.pop('type')
  return getattr(optim, t)(parameters, **config)


def get_scheduler(optimizer, config):
  config= config.copy()
  t = config.pop('type')
  return getattr(optim.lr_scheduler, t)(optimizer, **config)


class BinaryFocalLoss(nn.Module):
  """BinaryFocalLoss class.

  For binary focal loss.
  """
  def __init__(self, gamma=0):
    super(BinaryFocalLoss, self).__init__()
    self.gamma = gamma

  def forward(self, prob, target):
    """
    prob : [batch_size, num_classes]
    target : [batch_size, num_classes]
    """
    pt = (1 - target) * prob + target * (1 - prob)
    logpt = torch.log(pt)
    loss = -1 * (1 - pt) ** self.gamma * logpt
    return loss.mean()


def get_loss(config):
  """Return a callable object."""
  config = config.copy()
  t = config.pop('type')
  if t == 'BinaryFocalLoss':
    return BinaryFocalLoss(**config)
  else:
    return getattr(nn, t)(**config)


_CUSTOM_TRANSFORMS = {}
def register_custom_transform(cls):
  n = cls.__name__
  if n in _CUSTOM_TRANSFORMS:
    assert False
  _CUSTOM_TRANSFORMS[n] = cls
  return cls

def parse_config(config_file):
  if not os.path.isfile(config_file):
    raise FileNotFoundError("File %s don't exist" % config_file)
  with open(config_file, 'r') as f:
    content = f.read()
  __locals = {}
  exec("config = %s" % content, __locals)
  config = __locals['config']
  return config


class CustomTransform(object):
  def __call__(self, image):
    raise NotImplementedError()


@register_custom_transform
class CustomRandomGaussianNoise(CustomTransform):
  def __init__(self, mean=0, std=0.1):
    self.mean = mean
    self.std = std

  def __call__(self, image):
    img = np.array(image).astype(np.float32)
    noise = np.random.normal(self.mean, self.std, size=img.shape)
    img += noise
    img = PIL.Image.fromarray(np.uint8(img))
    return img


@register_custom_transform
class CustomRandomSPNoise(CustomTransform):
  def __init__(self, prob=0.1):
    self.prob = prob

  def __call__(self, image):
    img = np.array(image)
    noise = np.random.uniform(size=img.shape)
    img[noise < self.prob] = 0
    img = PIL.Image.fromarray(np.uint8(img))
    return img


def parse_transform(config):
  config = config.copy()
  t = config.pop('type')
  config = config['params']
  if config is None or t is None or len(config) == 0:
    return None
  res = []
  for trans in config:
    if not isinstance(trans, tuple):
      raise TypeError("Wrong type")
    if len(trans) != 2 or len(trans[1]) != 2:
      raise ValueError("Wrong format")
    assert isinstance(trans[1][0], list), trans[1][0]
    assert isinstance(trans[1][1], dict), trans[1][1]
    if trans[0].startswith('Custom'):
      res.append(_CUSTOM_TRANSFORMS[trans[0]](*trans[1][0], **trans[1][1]))
    else:
      res.append(getattr(transforms, trans[0])(*trans[1][0], **trans[1][1]))
  return getattr(transforms, t)(res)


def get_latest_ckpt(path):
  pat = re.compile("epoch=(\d+)")
  ps = os.listdir(path)
  ps = [os.path.basename(p) for p in ps if p.endswith('ckpt')]
  if len(ps) == 0:
    return None
  ps.sort(key=lambda a: int(pat.search(a).groups()[0]))
  return os.path.join(path, ps[-1])


def get_min_loss_ckpt(path):
  pat = re.compile("val_loss=(\d+\.\d+)")
  ps = os.listdir(path)
  ps = [os.path.basename(p) for p in ps if p.endswith('ckpt')]
  if len(ps) == 0:
    return None
  ps.sort(key=lambda a: float(pat.search(a).groups()[0]), reverse=True)
  return os.path.join(path, ps[-1])
