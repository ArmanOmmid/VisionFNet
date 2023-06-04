
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
        ('--experiment_path', experiment_path)
    ]
    arguments = [item for sublist in options for item in sublist] + unparsed_args
    command = ['python3', 'experiment.py'] + arguments

    process = Popen(command, stdout=PIPE, stderr=PIPE)

    command_string = " ".join(command) + '\n'
    terminal_file = open(terminal_path, 'w')
    terminal_file.write(command_string)

    stdout_lines = process.stdout.readlines()
    for line in stdout_lines:
        line = line.decode()
        terminal_file.write(line)
        print(line.strip('\n'))
    
    if process.stderr is not None:
        stderr_lines = process.stderr.readlines()
        if len(stderr_lines) != 0:
            error_message = "\n\n==== Unhandled Exception Encountered ====\n\n"
            print(error_message)
            for line in stderr_lines:
                line = line.decode()
                terminal_file.write(line)
                print(line.strip('\n'))

    process.stdout.close()
    if process.stderr is not None: process.stderr.close()

    terminal_file.close()

    return_code = process.wait()

    return return_code






if __name__ == "__main__":

    args, unkown_args = parser.parse_known_args()
    main(args, unkown_args)

    gc.collect()

