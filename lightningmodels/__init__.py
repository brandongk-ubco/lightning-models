import lightningdata  # noqa: F401
import pl_bolts  # noqa: F401
from . import vision
import sys
from functools import partial
from pytorch_lightning.utilities.cli import LightningCLI, DATAMODULE_REGISTRY, MODEL_REGISTRY
import os
import json
import importlib

modules_file = os.path.join(os.getcwd(), "modules.json")
if os.path.exists(modules_file):
    with open(modules_file, "r") as jsonfile:
        modules = json.load(jsonfile)
    if "models" in modules:
        for module in modules["models"]:
            importlib.import_module(module)


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes",
                              "model.num_classes",
                              apply_on="instantiate")
        parser.link_arguments("data.in_channels",
                              "model.in_channels",
                              apply_on="instantiate")
        parser.link_arguments("data.dataset_name",
                              "model.dataset_name",
                              apply_on="instantiate")
        parser.link_arguments("data.classes",
                              "model.classes",
                              apply_on="instantiate")


CLI = partial(MyLightningCLI,
              seed_everything_default=42,
              trainer_defaults={
                  "gpus": -1,
                  "deterministic": True,
                  "max_epochs": sys.maxsize
              })

usage = f"""
"usage: python -m lightningmodels {{fit,validate,test,predict,tune}} DATAMODULE MODEL [...]
positional arguments:
 DATAMODULE     a datamodule registered in pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY
                {', '.join(DATAMODULE_REGISTRY.keys())}
 MODEL          a model registered in pytorch_lightning.utilities.cli.MODEL_REGISTRY
                {', '.join(MODEL_REGISTRY.keys())}
"""


def trainer():
    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)
    dataset_name = sys.argv.pop(2)
    model = sys.argv.pop(2)
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Couldn't find model {model} in {', '.join(MODEL_REGISTRY.keys())}"
        )
    if dataset_name not in DATAMODULE_REGISTRY:
        raise ValueError(
            f"Couldn't find dataset {dataset_name} in {', '.join(DATAMODULE_REGISTRY.keys())}"
        )
    CLI(
        MODEL_REGISTRY[model],
        DATAMODULE_REGISTRY[dataset_name],
    )


__all__ = ["vision", "CLI", "execute"]
