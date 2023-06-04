
import __init__

import sys
import os
import time
import gc
import random
from collections import Counter
import argparse
import copy
import yaml
import datetime
import json

from subprocess import Popen, PIPE, STDOUT

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('-E', '--experiment_path', default=False,
                    help="Path to save experiment results")

def main(args, unparsed_args):

    experiment_path = args.experiment_path

    experiment_name = '_'.join(str(datetime.datetime.now()).split(' ')).split('.')[0]

    if experiment_path:
        if str(__init__.repository_root) in os.path.abspath(experiment_path):
            experiment_path = os.path.join(__init__.repository_root, 'experiments', experiment_name) # Always redirect plots to the designated plot folder if its in the repo
        else:
            os.makedirs(os.path.join(experiment_path, experiment_name), exist_ok=True)
            experiment_path = os.path.join(experiment_path, experiment_name)
    else:
        experiment_path = os.path.join(__init__.repository_root, 'experiments', experiment_name)

    terminal_path = os.path.join(experiment_path, 'terminal.txt')


    options = [
        ('--experiment_path', args.experiment_path),
        ('--experiment_name', experiment_name)
    ]
    arguments = [item for sublist in options for item in sublist] + unparsed_args
    main_program = os.path.join(os.path.dirname(__file__), 'experiment.py')
    command = ['python3', main_program] + arguments

    process = Popen(command, stdout=PIPE, stderr=PIPE)

    command_string = " ".join(command) + '\n'

    with open(terminal_path, 'wb') as terminal_file:

        terminal_file.write(command_string)

        for line in iter(process.stdout.readline, b""):
            terminal_file.write(line)
            sys.stdout.write(line)

        for line in iter(process.stderr.readline, b""):
            terminal_file.write(line)
            sys.stdout.write(line)

        return_code = process.wait()

    return return_code






if __name__ == "__main__":

    args, unkown_args = parser.parse_known_args()
    main(args, unkown_args)

    gc.collect()

