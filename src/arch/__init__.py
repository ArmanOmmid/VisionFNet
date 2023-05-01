
import sys
import os
repository_root = '/'.join(__file__.split('/')[:__file__.split('/').index('src')])
sys.path.append(repository_root)

files = [
    file for file in os.listdir(__file__) 
    if file.split('.') == 'py' and file not in ['__init__']
]

__all__ = files

