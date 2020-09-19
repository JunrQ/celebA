

# ------- #
# run.py  # 
# ------- #


from argparse import ArgumentParser
from pytorch_lightning import seed_everything

from utils import *
from models import CelebAModel
from trainer import get_trainer

parser = ArgumentParser()

parser.add_argument('--config', type=str, help='Config file path')
parser.add_argument('--path', type=str, help='Save path')
parser.add_argument('--gpus', type=str, help='Gpus used')
parser.add_argument('--eval', type=bool, help='Whether only do test')

# add model specific args
# parser = LitModel.add_model_specific_args(parser)
# parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
seed_everything(1234) # reproducibility


config = parse_config(args.config)
gpus = [int(x) for x in gpus.strip().split(',')]


criterion = get_loss(config['criterion'])
model = CelebAModel(criterion=criterion,
                    config=config,
                    batch_size=config['batch_size'],
                    **config['model'])
trainer = get_trainer(gpus=gpus,
                      path=args.path,
                      config['trainer'])
if args.eval:
  trainer.test(model)
else:
  trainer.fit(model)
