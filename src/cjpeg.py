from __future__ import print_function
from cjpegargs import _parser
from sys import argv, stdout
from glob import glob
import os
import sys
import fnmatch
import json

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
import cv2
from pkg_resources import resource_filename


_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
(_options, _args) = _parser.parse_args()


class FileNotFound(Exception):
    """docstring for FileNotFound"""

    def __init__(self, arg=''):
        super(FileNotFound, self).__init__('Figure not found: ' + arg)


def main():
    pass

if __name__ == '__main__':
    main()
