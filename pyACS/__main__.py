import argparse
from pyACS import __version__
from pyACS.acs import ConvertBinToCSV

# Argument Parser
parser = argparse.ArgumentParser(prog="python -m pyACS")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbosity.")
parser.add_argument("--version", action="version", version='pyACS v' + __version__,
                    help="Prints version information.")
parser.add_argument("dvc", type=str,
                    help="Device file.")
parser.add_argument("src", type=str,
                    help="Source file to decode and calibrate.")
parser.add_argument("dst", type=str, default=None,
                    help="Destination file of decoded and calibrated data.")
parser.add_argument("--auxiliaries", action="store_true",
                    help="Output auxiliary data (internal and external temperatures).")
args = parser.parse_args()

# Decode and Calibrate binary file
if args.verbose:
    print('Converting ' + args.src + ' ... ', end='', flush=True)
cbc = ConvertBinToCSV(args.dvc, args.src, args.dst, args.auxiliaries)
if args.verbose:
    print('Done')
    print('Frame extracted: ' + str(cbc.counter_good) + '\tFrame Incomplete: ' + str(cbc.counter_bad))
