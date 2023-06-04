
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

    options = [
        ('--experiment_path', experiment_path)
    ]

    arguments = [item for sublist in options for item in sublist] + unparsed_args
    command = ['python3', 'main'] + arguments

    process = Popen(command, stdout=PIPE, stderr=PIPE)

    command_string = " ".join(command) + '\n'

    stdout_lines = process.stdout.readlines()
    for line in stdout_lines():
        line = line.decode()
        print(line.strip('\n'))
    
    if process.stderr is not None:
        stderr_lines = process.stderr.readlines()
        if len(stderr_lines) != 0:
            error_message = "\n\n==== Unhandled Exception Encountered Encountered ====\n\n"
            print(error_message)
            for line in stderr_lines:
                line = line.decode()
                print(line.strip('\n'))

    process.stdout.close()
    if process.stderr is not None: process.stderr.close()

    return_code = process.wait()

    return return_code






if __name__ == "__main__":

    args, unkown_args = parser.parse_known_args()
    main(args, unkown_args)

    gc.collect()

