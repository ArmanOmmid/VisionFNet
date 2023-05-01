
import sys
import os
repository_root = '/'.join(__file__.split('/')[:__file__.split('/').index('src')])
sys.path.append(repository_root)

import importlib

this = sys.modules[__name__]

print(type(this), this)

dirname = os.path.dirname(__file__)

mod = importlib.import_module('basic_fcn')

modules = [
    (this, name, importlib.import_module(name)) for name in dirname
    if name != '__init__py' and name[:-3] != '.py'
]

map(setattr, modules)

# for module in os.listdir(dirname):
#     if module == '__init__.py' or module[-3:] != '.py':
#         continue
#     __import__(module[:-3], locals(), globals())
# del module

# __all__ = [
#     module for module in os.listdir(dirname)
#     if module != '__init__py' and module[:-3] != '.py'
# ]