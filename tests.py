#!/usr/bin/python

import glob
from sys import argv, stdout

from multiprocessing.pool import ThreadPool
import subprocess

size_block = [4, 8, 16, 32, 64]


def execute(filename):
    for size in size_block:
        cmd = ['cjpeg', '-q', '-n {}'.format(size),
               '--no-save', '{}'.format(filename)]

        try:
            process = subprocess.Popen(cmd)
            process.wait()
        except Exception:
            pass


def main():
    pattener = ["*.jpg", '*.jpeg', '*.tiff', '*.bmp']

    _MAX_THREADS = 10
    times = 10
    _pool = ThreadPool(processes=_MAX_THREADS)
    files = []
    root = argv[1] if len(argv) > 1 else '.'

    for pat in pattener:
        files += glob.glob('{}/**/{}'.format(root, pat), recursive=True)

    stdout.write('{} files found\n'.format(len(files)))

    if len(files) == 0:
        return

    for i in range(1, times):
        print('run {}/{}'.format(i, times))
        result = _pool.map_async(execute, files)
        result.wait()

    print(result)

if __name__ == '__main__':
    main()
