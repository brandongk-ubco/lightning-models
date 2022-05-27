import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import segmentation_models_pytorch as smp
import torchmetrics
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from ...helpers.classwise import classwise
from ...losses.FocalLoss import FocalLoss
from ...losses.TverskyLoss import TverskyLoss
from typing import List, Any

__all__ = ["Segmenter"]


@MODEL_REGISTRY
class Segmenter(LightningModule):

    def __init__(self,
                 num_classes: int,
                 classes: List[Any],
                 in_channels: int,
                 patience: int = 5,
                 learning_rate: float = 5e-3,
                 min_learning_rate: float = 5e-4,
                 weight_decay: float = 0.0,
                 focal_loss_multiplier: float = 1.0,
                 tversky_loss_multiplier: float = 1.0,
                 dataset_name: str = None):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(self.hparams.in_channels,
                                    track_running_stats=True),
            torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model(), torch.nn.Sigmoid())

        self.train_f1 = torchmetrics.F1Score(self.hparams.num_classes + 1,
                                             average="macro",
                                             mdmc_average='global')
        self.valid_f1 = torchmetrics.F1Score(self.hparams.num_classes + 1,
                                             average="macro",
                                             mdmc_average='samplewise')
        self.test_f1 = torchmetrics.F1Score(self.hparams.num_classes + 1,
                                            average="macro",
                                            mdmc_average='samplewise')

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
        ]

        try:
            callbacks.append(DeviceStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def loss(self, y_hat, y):
        assert y_hat.shape == y.shape

        focal_loss = classwise(
            y_hat,
            y,
            metric=lambda y_hat, y: -1
            if not y.max() > 0 else FocalLoss("binary", from_logits=False)
            (y_hat, y))

        focal_loss = self.hparams.focal_loss_multiplier * focal_loss[
            focal_loss >= 0].mean()

        tversky_loss = classwise(
            y_hat,
            y,
            metric=lambda y_hat, y: -1
            if not y.max() > 0 else TverskyLoss(from_logits=False)(y_hat, y))

        tversky_loss = self.hparams.tversky_loss_multiplier * tversky_loss[
            tversky_loss >= 0].mean()

        loss = focal_loss + tversky_loss
        return loss

    def get_model(self):
        return smp.Unet(encoder_name='efficientnet-b0',
                        encoder_weights="imagenet",
                        classes=self.hparams.num_classes)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        background = torch.ones(
            [y_hat.size(dim=0), 1,
             y_hat.size(dim=2),
             y_hat.size(dim=3)],
            dtype=y_hat.dtype,
            device=y_hat.device) * 0.5
        self.train_f1(
            torch.concat([background, y_hat], dim=1).argmax(dim=1),
            torch.concat([background, y], dim=1).argmax(dim=1))
        self.log('train_f1',
                 self.train_f1,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)

        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch

        y_hat = self(x)

        background = torch.ones(
            [y_hat.size(dim=0), 1,
             y_hat.size(dim=2),
             y_hat.size(dim=3)],
            dtype=y_hat.dtype,
            device=y_hat.device) * 0.5
        self.valid_f1(
            torch.concat([background, y_hat], dim=1).argmax(dim=1),
            torch.concat([background, y], dim=1).argmax(dim=1))
        self.log('valid_f1',
                 self.valid_f1,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        background = torch.ones(
            [y_hat.size(dim=0), 1,
             y_hat.size(dim=2),
             y_hat.size(dim=3)],
            dtype=y_hat.dtype,
            device=y_hat.device) * 0.5
        self.test_f1(
            torch.concat([background, y_hat], dim=1).argmax(dim=1),
            torch.concat([background, y], dim=1).argmax(dim=1))

        self.log('test_f1',
                 self.test_f1,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

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
