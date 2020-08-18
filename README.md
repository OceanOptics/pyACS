pyACS
=====
[![Python 3](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/downloads/)
[![license MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OceanOptics/pySatlantic/blob/master/LICENSE)

_Python package to unpack and calibrate binary data from a WetLabs ACS_

## Installation
The pyACS package can be installed directly with python's package manager.

    pip install pyACS

pyACS can also be installed from the sources available at [GitHub repository](https://github.com/OceanOptics/pyACS/).

    python setup.py install

Note that the only default dependency is numpy. Installing the library SciPy in addition will improve performances.
    
    pip install scipy
    
## Convert binary to CSV files
pyACS can be used from a terminal to convert binary files recorded with Compass software to CSV files. 

    python -m pyACS [-h] [-v] [--version] [-aux] device_file bin_file [destination]

positional arguments:
  device_file          Device file.
  bin_file             Source file to decode and calibrate (.bin).
  destination          Destination file of decoded and calibrated data.

optional arguments:
  -h, --help           Show this help message and exit
  -v, --verbose        Enable verbosity.
  --version            Prints version information.
  -aux, --auxiliaries  Output auxiliary data (internal and external temperatures).

## Embed in other Software
The class `ACS` provides key methods to handle the binary ACS data
* `read_device_file`: Parse device file to be able to unpack and calibrate binary frame
* `find_frame`: Find registered ACS frames in bytearray
* `unpack_frame`: Unpack/Decode a binary frame into named tuple `FrameContainer`
* `calibrate_frame`: Convert a frame engineering units (counts) into scientific units (1/m for a and c)

The classes `BinReader` and `ConvertBinToCSV` are an illustration of the usage of the ACS class to parse binary files recorded with Compass software (.bin).
