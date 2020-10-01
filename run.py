

# ------- #
# run.py  # 
# ------- #


from argparse import ArgumentParser
import os

from pytorch_lightning import seed_everything

from utils import *
from models import CelebAModel
from trainer import get_trainer

parser = ArgumentParser()

parser.add_argument('--config', type=str, help='Config file path')
parser.add_argument('--path', type=str, help='Save path')
parser.add_argument('--gpus', type=str, help='Gpus used')
parser.add_argument('--eval', type=bool, help='Whether only do test')

args = parser.parse_args()
seed_everything(1234) # reproducibility
debug = False


config = parse_config(args.config)
gpus = [int(x) for x in args.gpus.strip().split(',')]
if not os.path.isdir(args.path):
  os.mkdir(args.path)


criterion = get_loss(config['criterion'])
model = CelebAModel(criterion=criterion,
                    config=config,
                    path=args.path,
                    batch_size=config['batch_size'],
                    **config['model'])
trainer = get_trainer(gpus=gpus,
                      path=args.path,
                      debug=debug,
                      resume_mode='min_loss' if args.eval else 'latest',
                      config=config['trainer'])
if not args.eval:
  trainer.fit(model)
trainer.test(model)

