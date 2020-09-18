
import ArgumentParser


import torch
import torchvision.models as models



class CelebAModel(LightningModule):

  def __init__(self, ):
    super(CelebAModel, self).__init__()


    self.save_hyperparameters('layer_1_dim', 'learning_rate')



  def validation_step(self, batch, batch_idx):
    result = pl.EvalResult(checkpoint_on=loss)
    result.log('val_loss', loss)

    # equivalent
    result.log('val_loss', loss, prog_bar=False, logger=True,
               on_step=False, on_epoch=True, reduce_fx=torch.mean)


    x, y = batch
    y_hat = self.model(x)
    loss = F.cross_entropy(y_hat, y)
    acc = FM.accuracy(y_hat, y)
    result = pl.EvalResult(checkpoint_on=loss)
    result.log_dict({'val_acc': acc, 'val_loss': loss})

    return result


  def test_step(self, batch, batch_idx):
    result = self.validation_step(batch, batch_idx)
    result.rename_keys({'val_acc': 'test_acc', 'val_loss': 'test_loss'})


    return result

  
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)

    result = pl.TrainResult(loss)

    # logs metrics for each training_step, and the average across the epoch,
    # to the progress bar and logger
    result.log('train_loss', loss, on_step=True, on_epoch=True, 
               prog_bar=True, logger=True)
    return pl.TrainResult(loss)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.02)




  def train_dataloader(self):
    return DataLoader(mnist_train, batch_size=self.hparams.batch_size)



  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--encoder_layers', type=int, default=12)
    parser.add_argument('--data_path', type=str, default='/some/path')
    return parser




