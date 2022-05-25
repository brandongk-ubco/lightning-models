import pprint
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


def list():
    pprint.pprint(MODEL_REGISTRY)
