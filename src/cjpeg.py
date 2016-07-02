from __future__ import print_function
from sys import argv, stdout
from glob import glob
import os
import sys
import fnmatch
import json

from multiprocessing.pool import ThreadPool
from cjpegargs import _parser
from writebits import Bitset
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import numpy as np
import itertools

from pkg_resources import resource_filename
from qtables import huffman_dc, huffman_luminance
try:
    import cv2
except Exception:
    import local_cv2


_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
_options = _parser.parse_args()


def print(data, end='\n'):
    """overwrite print"""
    if _options.verbose:
        stdout.write(str(data) + end)


class FileNotFound(Exception):
    """docstring for FileNotFound exception"""

    def __init__(self, arg=''):
        if len(arg) > 1:
            output = "Figure not found or isn't a valid figure: "
        else:
            output = "Figure not found or isn't a valid figure."

        super(FileNotFound, self).__init__(output + arg)


class EmptyFile(Exception):
    """docstring for EmptyFile exception"""

    def __init__(self, arg=''):
        output = "There is nothing to fill the output"
        if arg:
            output += ': [{}]'.format(arg)
        super(EmptyFile, self).__init__(output)


class CustomJpeg(object):
    """docstring for CustomJpeg"""

    def __init__(self, filename):
        super(CustomJpeg, self).__init__()
        self.filename = filename
        # 0 is to read as grayscale
        self.figure = cv2.imread(self.filename, 0)
        # generate image RGB -> YCbCr
        self.figure_ycbcr = cv2.cvtColor(
            cv2.imread(self.filename), cv2.COLOR_BGR2YCR_CB)
        # got each element from YCbCr
        self.y, self.cb, self.cr = self._split_channel_(self.figure_ycbcr)

        if not _options.output:
            self.output_filename = self.filename.replace(
                self.filename.split('.')[-1], 'cjpeg')
        else:
            self.output_filename = _options.output
        if self.figure is None:
            raise FileNotFound(self.filename)
        self.shape = self.figure.shape
        self.pixs = _options.size
        self.scrambled = np.array([])

        self.bitarray = Bitset()
        self.bitarray.name = self.output_filename
        self.bitarray.verbose = False

    def encode(self, output=''):
        """encode de file"""
        # will be only one block
        if True in (self.pixs > np.array(self.figure.shape)):
            self.pixs = min(self.figure.shape)
            import warnings
            warnings.simplefilter("always")
            warnings.warn('size updated to {}'.format(
                self.pixs), Warning)
        self.blocks_split()
        self.output = self.scrambled.copy()
        for i in range(len(self.scrambled)):
            self.output[i] = CustomJpeg._customDCT_(self.scrambled[i])
        self.quantize()
        self.output = self._blocks_merge_(
            self.output, self.figure.shape, self.pixs)

    def save(self):
        # save
        if len(self.bitarray) > 0:
            self.bitarray.to_file()

    def size(self):
        return len(self.bitarray) / 8

    def quantize(self):
        """ Quantize the output to write into the file"""
        # start value to DC
        DC = 0
        for block in self.output:
            list_block = self.zig_zag(block)
            # [new DC] = [first] - [old DC]
            DC = list_block[0] - DC
            # lets format the list like the outpush shows in:
            # http://www.pcs-ip.eu/index.php/main/edu/8
            bits = []
            # first stage
            bits.append(DC)
            zero_counts = 0
            for value in list_block[1:]:
                if value == 0:
                    zero_counts += 1
                    if zero_counts == 10:
                        bits.append([zero_counts, value])
                        zero_counts = 0
                else:
                    bits.append([zero_counts, value])
                    zero_counts = 0

            # second stage
            bit_size = len('{:b}'.format(abs(bits[0])))
            bits[0] = [[bit_size], bits[0]]
            for bit in range(1, len(bits)):
                bit_size = len('{:b}'.format(abs(bits[bit][1])))
                bits[bit][0] = [bits[bit][0], bit_size]

            # third stage
            for bit in range(len(bits)):
                value = bits[bit][-1]
                # using U1
                if value < 0:
                    binary = '{:b}'.format(((1 << 16) + value) - 1)
                else:
                    binary = '{:b}'.format(value)
                # cut the string
                bit_size = bits[bit][0][-1]
                binary = binary[-bit_size:]
                bits[bit][-1] = binary

            # four stage (huffman)
            bits[0][0] = huffman_dc[bits[0][0][0]]
            self.bitarray.push(bits[0][0])
            self.bitarray.push(bits[0][1])
            for bit in range(1, len(bits)):
                bits[bit][0] = huffman_luminance[str(bits[bit][0])]
                self.bitarray.push(bits[bit][0])
                self.bitarray.push(bits[bit][1])

            #  its ready!

    def blocks_merge(self):
        """merge splited image into one"""

        self.figure = CustomJpeg._blocks_merge_(
            self.scrambled,
            self.figure.shape,
            self.pixs
        )

    def blocks_split(self):
        """split a image into NxN blocks. N=self.pixs"""
        self.scrambled = CustomJpeg._blocks_split_(
            self.figure, self.pixs)

    def show(self, name=''):
        """show the figure"""
        if name == '':
            name = self.output_filename
        cv2.imshow(name, self.figure)
        cv2.waitKey(0)

    def write(self):
        """write the bitarray to a file"""
        if not len(self.bitarray):
            raise EmptyFile(self.bitarray.name)
        self.bitarray.to_file()

    @staticmethod
    def _split_channel_(img):
        channels = []
        for ch in range(img.shape[-1]):
            channels.append(img[..., ch])
        return channels

    @staticmethod
    def _concatenate_channels_(ch1, ch2, ch3):
        assert ch1.ndim == 2 and ch2.ndim == 2 and ch3.ndim == 2
        rgb = (ch1[..., np.newaxis],
               ch2[..., np.newaxis], ch3[..., np.newaxis])
        return np.concatenate(rgb, axis=-1)

    @staticmethod
    def _customDCT_(block):
        """applying DCT in a macroblock"""
        imf = np.float32(block) / 255.0  # float conversion/scale
        dst = cv2.dct(imf)           # the dct
        img = np.uint8(dst) * 255.0

        return img

    @staticmethod
    def _blocks_merge_(scrambled, shape, pixs=8):

        no_rows, no_cols = shape

        rows_n = int((no_rows / pixs) * pixs)
        cols_n = int((no_cols / pixs) * pixs)
        figure = np.zeros(shape, scrambled.dtype)

        next_block = 0

        # reassemble the blocks
        for row in range(rows_n - pixs + 1):
            for col in range(cols_n - pixs + 1):
                if (row % pixs == 0 and col % pixs == 0):
                    cur_block = scrambled[next_block].copy()
                    next_block += 1
                    figure[row:row + pixs, col:col + pixs] = cur_block

        return figure

    @staticmethod
    def _blocks_split_(figure, pixs=8):
        scrambled = figure.copy()

        no_rows, no_cols = scrambled.shape

        rows_n = int((no_rows / pixs) * pixs)
        cols_n = int((no_cols / pixs) * pixs)

        scrambled = cv2.resize(scrambled, (cols_n, rows_n))

        allBlocks = []

        # for loops to extract all blocks
        for row in range(rows_n - pixs + 1):
            for col in range(cols_n - pixs + 1):
                if (row % pixs == 0 and col % pixs == 0):
                    block = scrambled[row:row + pixs, col:col + pixs].copy()
                    allBlocks.append(block)

        return np.array(allBlocks, dtype=figure.dtype)

    @staticmethod
    def zig_zag(figure):
        """return the zig-zag of a block
            don't edit original block
        """
        if figure.shape[0] != figure.shape[1]:
            raise Exception('Block sould be square')

        n = figure.shape[0]
        output = np.array([], dtype=figure.dtype)

        def move(i, j):
            """inside method"""
            if j < (n - 1):
                return max(0, i - 1), j + 1
            else:
                return i + 1, j

        x, y = 0, 0
        for v in figure.flatten():
            output = np.append(output, figure[y][x])
            if (x + y) & 1:
                x, y = move(x, y)
            else:
                y, x = move(y, x)

        return output


def main():
    if (not _options.filename):
        if _options.args:
            _options.filename = _options.args
        else:
            _parser.print_help()
            return

    cj = CustomJpeg(_options.filename)
    cj.encode()
    cj.save()

    if _options.verbose:
        windows = plt.figure()
        windows.add_subplot(1, 2, 1)
        plt.imshow(cj.figure, cmap='Greys_r')
        windows.add_subplot(1, 2, 2)
        plt.imshow(cj.output, cmap='Greys_r')
        plt.show()

    print('{} > {} ({:.2f}%)'.format(os.path.getsize(_options.filename),
                                     cj.size(),
                                     100 * (1 - cj.size() /
                                            os.path.getsize(_options.filename))
                                     ))

if __name__ == '__main__':
    main()
