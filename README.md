pyACS
=====

_Python package to unpack and calibrate binary data from a WetLabs ACS_

The ACS module can be loaded and used in a script or can be used as it stands to convert binary files recorded by Compass Software to CSV files.

## Installation
The package runs with python 3 only. Install it with PyPI using the command:

    pip install pyACS

Or from the sources available on [GitHub repository](https://github.com/OceanOptics/pyACS/).
The distribution archive can be generated using:

    python3 setup.py sdist bdist_wheel

## Convert binary to CSV files
The package can be used as it stands from the command line to convert binary files to 

    python -m pyACS [-h] [-v] [--version] [--auxiliaries] dvc src dst

positional arguments:
  dvc            Device file.
  src            Source file to decode and calibrate.
  dst            Destination file of decoded and calibrated data.

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  Enable verbosity.
  --version      Prints version information.
  --auxiliaries  Output auxiliary data (internal and external temperatures).

## Embed in other Software
The class `ACS` provides key methods to handle the binary ACS data
* `read_device_file`: Parse device file to be able to unpack and calibrate binary frame
* `unpack_frame`: Unpack/Decode binary frame into named tuple `FrameContainer`
* `calibrate_frame`: Convert engineering units (counts) into scientific units (1/m for a and c)

The class `BinReader` helps to separate individual frames looking the registration bytes `b'\xff\x00\xff\x00'`. An example of usage of that class is `ConvertBinToCSV` which converts a binary file recorded with Compass into a CSV file. 
