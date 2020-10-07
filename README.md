# AI6126 Project 1: CelebA Facial Attribute Recognition Challenge

**Required packages**

* pytorch-lighting
* pytorch
* pandas = 1.1.1 *required by dataset.CelebA*
* requests = 2.24.0
* tqdm = 4.49.0

## File descriptions

* `models.py` : Define the model architeture and training, evaluation operation.
* `trainer.py` : Define trainer.
* `nets.py` : Some backbones.
* `run.py` : Script used for running the training and evaluation.
* `utils.py` : Define some functions.

## Run

In order to train or test a model, using the following script

```shell
# Specify the configuration used
config=v1.0.1.config

# Specify gpus
gpus=0

# Run
python -u run.py \
  --config config/${config} \
  --path log/${config} \
  --gpus ${gpus} \
  > log/${config}.log 2>&1
```

## Experiments Results

| version | Test Accuracy |
|-----------|-------------|
| v0.1.config | 0.9058 |
| v0.6.config | 0.9064 |
| v0.7.config | 0.9100 |
| v0.7.1.config | 0.9115 |
| v0.7.2.config | 0.9172 |
| v1.0.0.config | 0.9043 |

