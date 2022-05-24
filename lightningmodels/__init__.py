import lightningdata  # noqa: F401
from . import vision
from pytorch_lightning.utilities.cli import LightningCLI
import sys
from functools import partial
from pytorch_lightning.utilities.cli import LightningCLI, DATAMODULE_REGISTRY, MODEL_REGISTRY


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes",
                              "model.num_classes",
                              apply_on="instantiate")
        parser.link_arguments("data.in_channels",
                              "model.in_channels",
                              apply_on="instantiate")
        parser.link_arguments("data.name",
                              "model.dataset",
                              apply_on="instantiate")


CLI = partial(MyLightningCLI,
              seed_everything_default=42,
              trainer_defaults={
                  "gpus": -1,
                  "deterministic": True,
                  "max_epochs": sys.maxsize,
                  "stochastic_weight_avg": True
              })

usage = f"""
"usage: python -m lightningmodels {{fit,validate,test,predict,tune}} DATAMODULE MODEL [...]
positional arguments:
 DATAMODULE     a datamodule registered in pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY
                {', '.join(DATAMODULE_REGISTRY.keys())}
 MODEL          a model registered in pytorch_lightning.utilities.cli.MODEL_REGISTRY
                {', '.join(MODEL_REGISTRY.keys())}
"""


def execute():
    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)
    dataset = sys.argv.pop(2)
    model = sys.argv.pop(2)
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Couldn't find model {model} in {', '.join(MODEL_REGISTRY.keys())}"
        )
    if dataset not in DATAMODULE_REGISTRY:
        raise ValueError(
            f"Couldn't find dataset {dataset} in {', '.join(DATAMODULE_REGISTRY.keys())}"
        )
    CLI(
        MODEL_REGISTRY[model],
        DATAMODULE_REGISTRY[dataset],
    )


__all__ = ["vision", "CLI", "execute"]
