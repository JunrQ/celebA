import os

import torch
import torch.nn as nn
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import torch.optim as optim


def get_dataset(path, transform, name='train', target_type='attr'):
  transform = parse_transform(transform)
  return CelebA(root=path, split=name, target_type=target_type,
                transform=transform, download=True)


def get_optimizer(config, parameters):
  config = config.copy()
  t = config.pop('type')
  return getattr(optim, t)(parameters, **config)


def get_scheduler(optimizer, config):
  config= config.copy()
  t = config.pop('type')
  return getattr(optim.lr_scheduler, t)(optimizer, **config)


def get_loss(config):
  """Return a callable object."""
  config = config.copy()
  t = config.pop('type')
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

