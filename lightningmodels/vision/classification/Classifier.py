import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor, BasePredictionWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import monai
import torchmetrics
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from typing import List, Any
import os
from functools import partial

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
                 min_learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 dataset_name: str = None,
                 use_softmax: bool = True,
                 sigmoid_epochs: int = 6):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(self.hparams.in_channels,
                                    track_running_stats=True),
            torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model())

        self.loss = torch.nn.BCELoss()

        self.valid_f1 = torchmetrics.F1Score(num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(num_classes=num_classes)

        self.final_activation = torch.sigmoid

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=3 * self.hparams.patience,
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

        if self.hparams.use_softmax and self.current_epoch >= self.hparams.sigmoid_epochs and self.final_activation == torch.sigmoid:
            self.final_activation = partial(torch.softmax, dim=1)
            print("Switching to Softmax Activation")

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        return loss

    def forward(self, x):
        y_hat = self.model(x)
        y_hat = self.final_activation(y_hat)
        return y_hat

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        predicted_class = y_hat.argmax(dim=1)
        actual_class = y.argmax(dim=1)

        self.valid_f1(predicted_class, actual_class)

        loss = self.loss(y_hat, y)

        self.log_dict({
            'valid_f1': self.valid_f1,
            'val_loss': loss
        },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

    def test_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        predicted_class = y_hat.argmax(dim=1)
        actual_class = y.argmax(dim=1)

        self.test_f1(predicted_class, actual_class)
        loss = self.loss(y_hat, y)

        self.log_dict({
            'test_f1': self.test_f1,
            'test_loss': loss
        },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

    def predict_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        predicted = y_hat.argmax(dim=1)
        ground_truth = batch[1].argmax(dim=1)

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
