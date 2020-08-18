import argparse
from pyACS import __version__
from pyACS.acs import ConvertBinToCSV

# Argument Parser
parser = argparse.ArgumentParser(prog="python -m pyACS")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbosity.")
parser.add_argument("--version", action="version", version='pyACS v' + __version__,
                    help="Prints version information.")
parser.add_argument("device_file", type=str,
                    help="Device file.")
parser.add_argument("bin_file", type=str,
                    help="Source file to decode and calibrate (.bin).")
parser.add_argument("destination", type=str, default=None, nargs='?',
                    help="Destination file of decoded and calibrated data.")
parser.add_argument("-aux", "--auxiliaries", action="store_true",
                    help="Output auxiliary data (internal and external temperatures).")
args = parser.parse_args()

# Decode and Calibrate binary file
if args.verbose:
    print('Unpacking and calibrating ' + args.bin_file + ' ... ', end='', flush=True)
cbc = ConvertBinToCSV(args.device_file, args.bin_file, args.destination, args.auxiliaries)
if args.verbose:
    print('DONE')
    print('\tFrames extracted: ' + str(cbc.counter_good))
    if cbc.counter_bad:
        print('\tInvalid frame(s): ' + str(cbc.counter_bad))
