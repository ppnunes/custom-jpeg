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
        if len(arg) > 1:
            output = "Figure not found or isn't a valid figure: "
        else:
            output = "Figure not found or isn't a valid figure."

        super(FileNotFound, self).__init__(output + arg)


class CustomJpeg(object):
    """docstring for CustomJpeg"""

    def __init__(self, filename):
        super(CustomJpeg, self).__init__()
        self.filename = filename
        # 0 is to read as grayscale
        self.figure = cv2.imread(self.filename, 0)
        if self.figure is None:
            raise FileNotFound(self.filename)


def main():
    if (not _options.filename):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return

    CustomJpeg(_options.filename)

if __name__ == '__main__':
    main()
