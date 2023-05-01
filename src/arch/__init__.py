
import sys
import os
repository_root = '/'.join(__file__.split('/')[:__file__.split('/').index('src')])
sys.path.append(repository_root)

dirname = os.path.dirname(__file__)
__all__ = [
    module for module in os.listdir(dirname)
    if module != '__init__py' or module[:-3] != '.py'
]

map(__import__, __all__)

# for module in os.listdir(dirname):
#     if module == '__init__.py' or module[-3:] != '.py':
#         continue
#     __import__(module[:-3], locals(), globals())
# del module