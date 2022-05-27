import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import monai
import torchmetrics
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from typing import List, Any
from pytorch_lightning.callbacks import BasePredictionWriter
import os

__all__ = ["Classifier"]


class ClassifierPredictionWriter(BasePredictionWriter):

    def __init__(self,
                 output_dir: str,
                 write_interval: str,
                 classes: List[Any] = None):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "predictions.csv")
        self.classes = classes
        with open(self.output_file, "w", encoding='utf8') as f:
            f.write("batch_idx,ground_truth,predicted\n")

    def write_on_batch_end(self, trainer, pl_module: 'LightningModule',
                           prediction: Any, batch_indices: List[int],
                           batch: Any, batch_idx: int, dataloader_idx: int):

        predicted = prediction["predicted"]
        ground_truth = prediction["ground_truth"]
        if self.classes is not None:
            predicted = self.classes[predicted]
            ground_truth = self.classes[ground_truth]
        with open(self.output_file, "a", encoding='utf8') as f:
            f.write(f"{batch_idx},{ground_truth},{predicted}\n")


@MODEL_REGISTRY
class Classifier(LightningModule):

    def __init__(self,
                 num_classes: int,
                 classes: List,
                 in_channels: int,
                 patience: int = 3,
                 learning_rate: float = 5e-3,
                 min_learning_rate: float = 5e-4,
                 weight_decay: float = 0.0,
                 dataset_name: str = None):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(self.hparams.in_channels,
                                    track_running_stats=True),
            torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model(),
            torch.nn.Softmax(dim=1))

        self.loss = torch.nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=2 * self.hparams.patience,
                          monitor='val_loss',
                          verbose=True,
                          mode='min'),
            ModelCheckpoint(monitor='val_loss',
                            save_top_k=1,
                            mode="min",
                            filename='{epoch}-{val_loss:.6f}'),
            ClassifierPredictionWriter(os.getcwd(),
                                       write_interval="batch",
                                       classes=self.hparams.classes)
        ]

        try:
            callbacks.append(DeviceStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def get_model(self):
        return monai.networks.nets.EfficientNetBN(
            "efficientnet-b0",
            spatial_dims=2,
            pretrained=True,
            num_classes=self.hparams.num_classes)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.train_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('train_acc',
                 self.train_acc,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)

        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.valid_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('valid_acc',
                 self.valid_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('test_acc',
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        predicted = y_hat.squeeze().argmax()
        ground_truth = batch[1].squeeze().argmax()
        import pdb
        pdb.set_trace()
        return {"predicted": predicted, "ground_truth": ground_truth}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_learning_rate,
            verbose=True,
            mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
