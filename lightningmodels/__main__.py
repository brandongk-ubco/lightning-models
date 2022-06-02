import argh
from lightningmodels import commands
from inspect import getmembers, isfunction
from lightningmodels import trainer
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trainer":
        sys.argv.pop(1)
        trainer()
    else:
        command_list = [o[1] for o in getmembers(commands) if isfunction(o[1])]
        parser = argh.ArghParser()
        parser.add_commands(command_list)
        parser.dispatch()
