
import sys
import os
repository_root = '/'.join(__file__.split('/')[:__file__.split('/').index('src')])
sys.path.append(repository_root)

dirname = os.path.dirname(__file__)

files = [
    file for file in os.listdir(dirname) 
    if file.split('.') == 'py' and file not in ['__init__.py']
]

__all__ = files

