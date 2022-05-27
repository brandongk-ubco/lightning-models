import argh
from lightningmodels import commands
from inspect import getmembers, isfunction
import os
import json
import importlib
from lightningmodels import trainer
import sys

if __name__ == "__main__":
    modules_file = os.path.join(os.getcwd(), "modules.json")
    if os.path.exists(modules_file):
        with open(modules_file, "r") as jsonfile:
            modules = json.load(jsonfile)
        if "models" in modules:
            for module in modules["models"]:
                importlib.import_module(module)

    if len(sys.argv) > 1 and sys.argv[1] == "trainer":
        sys.argv.pop(1)
        trainer()
    else:
        command_list = [o[1] for o in getmembers(commands) if isfunction(o[1])]
        parser = argh.ArghParser()
        parser.add_commands(command_list)
        parser.dispatch()
