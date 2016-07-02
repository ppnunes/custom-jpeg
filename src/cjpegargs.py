import optparse
import argparse


_parser = argparse.ArgumentParser(
    usage="""%(prog)s [OPTIONS]
Examples:

Encode a file using the algorithm:
    %(prog)s <file> [FLAGS]
    
Decode a file using the algorithm:
    %(prog)s <file>.cjpeg [FLAGS]
        """,
    description="Encode/Decode a file using the custom jpeg algorithm",
)

# quiet options
_parser.add_argument("-q", "--quiet",
                     dest="verbose",
                     action="store_false",
                     help="suppress non error messages",
                     default=True
                     )

_parser.add_argument("-f", "--filename",
                     dest="filename",
                     type=str,
                     help="Name of the file",
                     )

_parser.add_argument("-e", "--encode",
                     dest="encode",
                     action="store_true",
                     default=True,
                     help="Encode a file",
                     )

_parser.add_argument("-d", "--decode",
                     dest="encode",
                     action="store_false",
                     default=True,
                     help="Decode a file",
                     )

_parser.add_argument("-n", "--size",
                     dest="size",
                     type=int,
                     default=8,
                     help="The size of the blocks to encode/decode",
                     )

_parser.add_argument("-o", "--output",
                     dest="output",
                     type=str,
                     help="Output filename",
                     )

_parser.add_argument("--no-save",
                     dest="save",
                     default=True,
                     action="store_false",
                     help="Create the file",
                     )

_parser.add_argument("args", help="display a square of a given number",
                     type=str)
