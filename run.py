

# ------- #
# run.py  # 
# ------- #

from utils import *


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('--config', type=str, default='some_name')


# add model specific args
parser = LitModel.add_model_specific_args(parser)

parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()



config = parse_config(args.config)

train_dataset = get_dataset(name='train', **config['train_dataset'])
val_dataset = get_dataset(name='valid', **config['valid_dataset'])
test_dataset = get_dataset(name='test', **config['test_dataset'])





