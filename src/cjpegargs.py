import optparse


_parser = optparse.OptionParser(
    usage="""%prog [OPTIONS]
Examples:

Encode a file using the algorithm:
    cjpeg <file> [FLAGS]
    
Decode a file using the algorithm:
    cjpeg <file>.cjpeg [FLAGS]
        """,
    description="Encode/Decode a file using the custom jpeg algorithm",
)

# quiet options
_parser.add_option("-q", "--quiet",
                   dest="verbose",
                   action="store_false",
                   help="suppress non error messages",
                   default=True
                   )

_parser.add_option("-f", "--filename",
                   dest="filename",
                   type='string',
                   help="Name of the file",
                   )

_parser.add_option("-e", "--encode",
                   dest="encode",
                   action="store_true",
                   default=True,
                   help="Encode a file",
                   )

_parser.add_option("-d", "--decode",
                   dest="encode",
                   action="store_false",
                   default=True,
                   help="Decode a file",
                   )

_parser.add_option("-n", "--size",
                   dest="size",
                   type='int',
                   default=8,
                   help="The size of the blocks to encode/decode",
                   )
