import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor, BasePredictionWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import monai
import torchmetrics
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from typing import List, Any
import os
from bisect import bisect_right
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, SequentialLR
from functools import partial
import torchvision
from torch import nn

__all__ = ["Classifier"]


class SequentialLRWithMetrics(SequentialLR, ReduceLROnPlateau):

    def step(self, metrics, epoch=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]

        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler._reset()
                scheduler.step(metrics)
                self._last_lr = None
            else:
                scheduler.step(0)
                self._last_lr = scheduler.get_last_lr()
        else:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics)
                self._last_lr = None
            else:
                scheduler.step()
                self._last_lr = scheduler.get_last_lr()


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
                 patience: int = 5,
                 momentum: float = 0.9,
                 max_learning_rate: float = 0.1,
                 learning_rate: float = 1e-3,
                 min_learning_rate: float = 1e-5,
                 learning_rate_warmup_epochs: int = 30,
                 learning_rate_reduction_factor: float = 0.1,
                 weight_decay: float = 5e-4,
                 dataset_name: str = None,
                 final_activation: str = "sigmoid"):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.in_channels == 3:
            self.model = self.get_model()
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
                self.get_model())

        if self.hparams.final_activation == "softmax":
            self.loss = torch.nn.CrossEntropyLoss()
            self.final_activation = None
        elif self.hparams.final_activation == "normalized":
            self.loss = torch.nn.MSELoss()
            self.final_activation = partial(torch.nn.functional.normalize,
                                            dim=1)
        elif self.hparams.final_activation == "normalized_sigmoid":
            self.loss = torch.nn.BCELoss()
            self.final_activation = lambda input: torch.sigmoid(
                torch.nn.functional.normalize(input, dim=1))
        elif self.hparams.final_activation == "sigmoid":
            self.loss = torch.nn.BCELoss()
            self.final_activation = torch.sigmoid
        elif self.hparams.final_activation == "identity":
            self.loss = torch.nn.MSELoss()
            self.final_activation = None
        else:
            raise ValueError(
                "Must Specify Activation Function from {softmax, normalized, sigmoid, normalized_sigmoid, identity}"
            )

        self.valid_f1 = torchmetrics.F1Score(
            num_classes=self.hparams.num_classes)
        self.test_f1 = torchmetrics.F1Score(
            num_classes=self.hparams.num_classes)

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
        model = torchvision.models.resnet18(
            pretrained=False, num_classes=self.hparams.num_classes)
        model.conv1 = nn.Conv2d(3,
                                64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.maxpool = nn.Identity()
        return model

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        return loss

    def forward(self, x):
        y_hat = self.model(x)
        if self.final_activation:
            y_hat = self.final_activation(y_hat)
        return y_hat

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        actual_class = y.argmax(dim=1)
        y_hat = self(x)

        predicted_class = y_hat.argmax(dim=1)

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

        actual_class = y.argmax(dim=1)
        y_hat = self(x)
        predicted_class = y_hat.argmax(dim=1)

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

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    nesterov=True,
                                    weight_decay=self.hparams.weight_decay)

        warmup_scheduler = OneCycleLR(
            optimizer,
            div_factor=self.hparams.max_learning_rate /
            self.hparams.min_learning_rate,
            final_div_factor=self.hparams.min_learning_rate /
            self.hparams.learning_rate,
            max_lr=self.hparams.max_learning_rate,
            total_steps=self.hparams.learning_rate_warmup_epochs)

        reduction_scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_learning_rate,
            factor=self.hparams.learning_rate_reduction_factor,
            verbose=True,
            mode="min")

        scheduler = SequentialLRWithMetrics(
            optimizer,
            schedulers=[warmup_scheduler, reduction_scheduler],
            milestones=[self.hparams.learning_rate_warmup_epochs])

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

        return optimizer_config
