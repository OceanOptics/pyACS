from __future__ import print_function

import numpy as np
from math import log
from collections import namedtuple
from struct import unpack, unpack_from, calcsize
import csv
from sys import version_info, exit
try:
    from scipy import interpolate
except ImportError:
    SCIPY_IMPORTED = False
else:
    SCIPY_IMPORTED = True

# Check Python version running script
if version_info.major != 3:
    print("Script incompatible with Python {:d}.{:d}.{:d}".format(version_info.major,
                                                                  version_info.minor,
                                                                  version_info.micro))
    print("Please use Python 3.")
    exit(-1)


# Error Management
class ACSError(Exception):
    pass


class FrameIncompleteError(ACSError):
    pass


class FrameHeaderIncompleteError(FrameIncompleteError):
    pass


class NumberWavelengthIncorrectError(ACSError):
    pass


class FrameTypeIncorrectError(ACSError):
    pass


class FrameChecksumError(ACSError):
    pass


class SerialNumberIncorrectError(ACSError):
    pass


class BinReader:
    REGISTRATION = b'\xff\x00\xff\x00'
    READ_SIZE = 1024

    def __init__(self, filename=None):
        self.buffer = bytearray()
        if filename:
            self.run(filename)

    def run(self, filename):
        with open(filename, 'rb') as f:
            data = f.read(self.READ_SIZE)
            while data:
                self.data_read(data)
                data = f.read(self.READ_SIZE)
            self.handle_last_frame(self.REGISTRATION + self.buffer)  # KEEP REGISTRATION BYTE

    def data_read(self, data):
        self.buffer.extend(data)
        while self.REGISTRATION in self.buffer:
            frame, self.buffer = self.buffer.split(self.REGISTRATION, 1)
            if frame:
                self.handle_frame(self.REGISTRATION + frame)  # KEEP REGISTRATION BYTE

    def handle_frame(self, frame):
        raise NotImplementedError('Implement functionality in handle packet')

    def handle_last_frame(self, frame):
        return self.handle_frame(frame)


class CSVWriter:

    def __init__(self, lambda_c=None, lambda_a=None, write_auxiliaries=False):
        # Wavelength (in nm)
        self.lambda_c = lambda_c
        self.lambda_a = lambda_a

        self.write_auxiliaries = write_auxiliaries

        self.f = None
        self.writer = None

    def open(self, filename):
        fieldnames = ['timestamp'] + ['c%3.1f' % x for x in self.lambda_c] + ['a%3.1f' % x for x in self.lambda_a]
        if self.write_auxiliaries:
            fieldnames.extend(['temperature_internal', 'temperature_external'])
        self.f = open(filename, 'w')
        self.writer = csv.writer(self.f)  # , fieldnames=fieldnames)
        # self.writer.writeheader()
        self.writer.writerow(fieldnames)

    def write(self, raw, cal):
        if self.write_auxiliaries:
            self.writer.writerow([raw.time_stamp]
                                 + ["%.10f" % v for v in list(cal[0])]
                                 + ["%.10f" % v for v in list(cal[1])]
                                 + ["%.6f" % cal[2]]
                                 + ["%.6f" % cal[3]])
        else:
            self.writer.writerow([raw.time_stamp]
                                 + ["%.10f" % v for v in list(cal[0])]
                                 + ["%.10f" % v for v in list(cal[1])])

    def close(self):
        self.f.close()
    # TODO: Add test


class ConvertBinToCSV(BinReader):

    def __init__(self, device_filename, bin_filename, csv_filename=None, write_auxiliaries=False):
        if not csv_filename:
            csv_filename = bin_filename + '.dat'
        self.calibrate_auxiliaries = write_auxiliaries
        self.counter_good = 0
        self.counter_bad = 0
        self.acs = ACS(device_filename)
        self.csv = CSVWriter(self.acs.lambda_c, self.acs.lambda_a, write_auxiliaries)
        self.csv.open(csv_filename)
        super(ConvertBinToCSV, self).__init__(bin_filename)

    def handle_frame(self, frame):
        try:
            raw_frame = self.acs.unpack_frame(frame)
            cal_frame = self.acs.calibrate_frame(raw_frame, self.calibrate_auxiliaries)
            self.counter_good += 1
            self.csv.write(raw_frame, cal_frame)
        except FrameIncompleteError:
            self.counter_bad += 1


def compute_external_temperature(counts):
    # Convert external temperature engineering units (counts) to scientific units (deg C)
    return -7.1023317e-13 * counts ** 3 + \
           7.09341920e-8 * counts ** 2 + \
           -3.87065673e-3 * counts + 95.8241397


def compute_internal_temperature(counts):
    # Convert internal temperature engineering units (counts) to scientific units (deg C)
    volts = 5 * counts / 65535
    resistance = 10000 * volts / (4.516 - volts)
    return 1 / (0.00093135 + 0.000221631 * log(resistance) + 0.000000125741 * log(resistance) ** 3) - 273.15


def compute_checksum(frame):
    # Compute sum of all individual bytes as 2 bytes
    #   from registering packet (included) to checksum (excluded)
    return np.uint16(sum(frame[0:-3]))


FrameContainer = namedtuple('FrameContainer', ['frame_len', 'frame_type', 'serial_number',
                                               'a_ref_dark', 'p', 'a_sig_dark',
                                               't_ext', 't_int',
                                               'c_ref_dark', 'c_sig_dark',
                                               'time_stamp', 'output_wavelength',
                                               'c_ref', 'a_ref', 'c_sig', 'a_sig'])


class ACS:
    # Calibrate a and c raw values (sig, ref counts) to scientific units (1/m)

    FRAME_HEADER_FORMAT = '!HBBlHHHHHHHIBB'
    FRAME_HEADER_LEN = calcsize(FRAME_HEADER_FORMAT)
    FRAME_BACKER_FORMAT = '!Hc'
    FRAME_BACKER_LEN = calcsize(FRAME_BACKER_FORMAT)
    FRAME_CORE_OFFSET = FRAME_HEADER_LEN + 4
    FRAME_CHECKSUM_OFFSET = None

    def __init__(self, device_filename=None):
        # Meta data
        self.serial_number = None   # and Meter Type
        self.structure_version_number = None
        self.baudrate = 115200
        self.output_wavelength = None

        # Depth calibration
        self.depth_cal_scale_factor = None
        self.depth_cal_offset = None

        # Path length (in m)
        self.x = 0.25

        # Wavelength (in nm, not used)
        self.lambda_c = None
        self.lambda_a = None

        # Water offset value (in 1/m)
        self.offset_c = None
        self.offset_a = None

        # Internal temperature compensation value (in 1/m)
        self.t = None
        self.delta_t_c = None
        self.delta_t_a = None
        self.f_delta_t_c = None
        self.f_delta_t_a = None

        # Unpack frame format
        self.frame_core_format = None

        if device_filename:
            self.read_device_file(device_filename)

    def __repr__(self):
        return 'SN: ' + self.serial_number + '\n' + \
               'V: ' + str(self.structure_version_number) + '\n' + \
               'BD: ' + str(self.baudrate) + '\n' + \
               'WL: ' + str(self.output_wavelength) + '\n' + \
               'WL(c): ' + str(self.lambda_c.size) + ' ' + str(self.lambda_c) + '\n' + \
               'WL(a): ' + str(self.lambda_a.size) + ' ' + str(self.lambda_a) + '\n' + \
               'T:' + str(self.t.size) + ' ' + str(self.t) + '\n' + \
               'DT(c): ' + str(np.size(self.delta_t_c, 0)) + ' ' + str(np.size(self.delta_t_c, 1)) + ' '\
                         + str(self.delta_t_c) + '\n' + \
               'DT(a): ' + str(np.size(self.delta_t_a, 0)) + ' ' + str(np.size(self.delta_t_a, 1)) + ' '\
                         + str(self.delta_t_a) + '\n' + \
               'WO(c): ' + str(self.offset_c.size) + ' ' + str(self.offset_c) + '\n' + \
               'WO(a): ' + str(self.offset_a.size) + ' ' + str(self.offset_a) + '\n' + \
               'FMT_F: ' + str(self.frame_core_format) + '\n'

    def read_device_file(self, filename):
        with open(filename, 'r') as f:
            iwl = 0  # line/index of wavelength
            for l in f:
                if 'Serial number' in l:
                    self.serial_number = '0x' + l.split(';')[0].strip('\t').lower()
                elif 'structure version number' in l:
                    self.structure_version_number = int(l.split(';')[0])
                elif 'Depth calibration' in l:
                    foo = l.split(';')[0].split('\t')
                    self.depth_cal_offset = float(foo[0])
                    self.depth_cal_scale_factor = float(foo[1])
                elif 'Baud rate' in l:
                    self.baudrate = int(l.split(';')[0])
                elif 'Path length' in l:
                    self.x = float(l.split(';')[0])
                elif 'output wavelengths' in l:
                    self.set_output_wavelength(int(l.split(';')[0]))
                    self.offset_c = np.empty(self.output_wavelength)
                    self.offset_a = np.empty(self.output_wavelength)
                    self.lambda_c = np.empty(self.output_wavelength)
                    self.lambda_a = np.empty(self.output_wavelength)
                    if self.t:
                        self.delta_t_c = np.empty((self.output_wavelength, self.t.size))
                        self.delta_t_a = np.empty((self.output_wavelength, self.t.size))
                elif 'number of temperature bins' in l:
                    n = int(l.split(';')[0])
                    if self.output_wavelength:
                        self.delta_t_c = np.empty((self.output_wavelength, n))
                        self.delta_t_a = np.empty((self.output_wavelength, n))
                elif l[0] == '\t':
                    # Temperatures
                    self.t = np.array(l.split(';')[0].strip().split('\t')).astype(np.float)
                elif l[0] == 'C':
                    foo = l.split('\t\t')
                    # Wavelength
                    bar = foo[0].split('\t')
                    self.lambda_c[iwl] = float(bar[0][1:])
                    self.lambda_a[iwl] = float(bar[1][1:])
                    # Water offset
                    self.offset_c[iwl] = float(bar[3])
                    self.offset_a[iwl] = float(bar[4])
                    # Internal temperature compensation
                    self.delta_t_c[iwl, :] = np.array(foo[1].split('\t'))
                    self.delta_t_a[iwl, :] = np.array(foo[2].split('\t'))
                    iwl += 1
                # skip lines "ACS Meter", "tcal[...]", and "maxANoise	maxCNoise[...]"
            if SCIPY_IMPORTED:
                # Build 2D interpolation function for speed
                self.f_delta_t_c = interpolate.interp1d(self.t, self.delta_t_c, axis=1, assume_sorted=True, copy=False,
                                     bounds_error=False, fill_value=(self.delta_t_c[:,1], self.delta_t_c[:,-1]))
                self.f_delta_t_a = interpolate.interp1d(self.t, self.delta_t_a, axis=1, assume_sorted=True, copy=False,
                                     bounds_error=False, fill_value=(self.delta_t_a[:,1], self.delta_t_a[:,-1]))
            #TODO: Add test

    def set_output_wavelength(self, output_wavelength):
        # Change number of wavelength of object
        #   adjust frame format for unpack
        #   update variable output_wavelength value

        # Make string format to unpack the binary frame
        #   The frame format and length is a function of the number of wavelength
        #   Comments are directly extracted from documentation
        # Make unpack format to get data
        # # Use network format (same as big-endian but mention that it will go over network)
        # fmt = '!'
        # # 2 bytes: packet length
        # fmt += 'H'
        # # 1 byte: Packet type identifier
        # fmt += 'B'
        # # 1 byte: reserved for future use
        # fmt += 'B'
        # # 4 bytes long integer: Meter Type + Instrument Serial Number
        # fmt += 'L'
        # # 2 bytes: A reference dark counts (for diagnostic purpose)
        # fmt += 'H'
        # # 2 bytes: A/D counts from the pressure sensor circuitry
        # fmt += 'H'
        # # 2 bytes: A signal dark counts (for diagnostic purpose)
        # fmt += 'H'
        # # 2 bytes: External temperature voltage counts
        # fmt += 'H'
        # # 2 bytes unsigned integer:  Internal temperature voltage counts
        # fmt += 'H'
        # # 2 bytes: C reference dark counts
        # fmt += 'H'
        # # 2 bytes: C signal dark counts
        # fmt += 'H'
        # # 4 bytes unsigned integer: Time stamp (ms)
        # fmt += 'I'
        # # 1 byte: reserved for future use
        # fmt += 'B'
        # # 1 byte unsigned integer: Number of output wavelength
        # fmt += 'B'
        # 2 bytes unsigned: Data for scan (c_ref, a_ref, c_sig, a_sig ... )
        # for i in range(output_wavelength):
        #     self.frame_core_format += 'HHHH'
        # # 2 bytes: Check sum
        # fmt += 'H'
        # # 1 byte: Last character 0x00
        # fmt += 'c'

        # Set variable section of frame format function of number wavelength
        self.frame_core_format = '!'
        for i in range(output_wavelength):
            self.frame_core_format += 'HHHH'
        # Set number of output wavelength
        self.output_wavelength = output_wavelength
        # Adjust checksum offset
        self.FRAME_CHECKSUM_OFFSET = self.FRAME_CORE_OFFSET + 4 * 2 * output_wavelength

    def unpack_frame(self, frame, force=False):
        # Unpack binary frame into human readable values
        if len(frame) < self.FRAME_HEADER_LEN:
            raise FrameHeaderIncompleteError('Frame header incomplete.')

        # Get frame header (skip registration packet)
        d = unpack_from(self.FRAME_HEADER_FORMAT, frame, offset=4)
        data = FrameContainer(frame_len=d[0],  # packet length
                              frame_type=d[1],  # packet type identifier
                              # data[] = d[2] # reserved for future use (1)
                              serial_number=hex(d[3]),  # instrument Serial Number (Meter type (first 2 bytes))
                              a_ref_dark=d[4],  # A reference dark counts (for diagnostic purpose)
                              p=d[5],  # A/D counts from the pressure sensor circuitry
                              a_sig_dark=d[6],  # A signal dark counts (for diagnostic purpose)
                              t_ext=d[7],  # External temperature voltage counts
                              t_int=d[8],  # unsigned integer:  Internal temperature voltage counts
                              c_ref_dark=d[9],  # C reference dark counts
                              c_sig_dark=d[10],  # C signal dark counts
                              time_stamp=d[11],  # unsigned integer: Time stamp (ms)
                              # data[] = d[12] # reserved for future use
                              output_wavelength=d[13], # number of output wavelength
                              c_ref=np.empty(d[13], dtype=np.uint16), a_ref=np.empty(d[13], dtype=np.uint16),
                              c_sig=np.empty(d[13], dtype=np.uint16),a_sig=np.empty(d[13], dtype=np.uint16))

        if force:
            if data.output_wavelength != self.output_wavelength:
                self.set_output_wavelength(data.output_wavelength)
        else:
            if data.frame_len != len(frame[0:-3]):
                # The frame length does not match the length indicated in the header of the packet received
                raise FrameIncompleteError('Frame incomplete.')
            if data.output_wavelength != self.output_wavelength:
                raise NumberWavelengthIncorrectError('Number of wavelength incorrect.')
            if data.serial_number != self.serial_number:
                raise SerialNumberIncorrectError('Serial number incorrect.')
            if data.frame_type < 3:  # 3 or higher for AC-S
                raise FrameTypeIncorrectError('Frame type incorrect (not AC-S).')
            # Check checksum
            checksum = unpack_from(self.FRAME_BACKER_FORMAT, frame,
                                   offset=self.FRAME_CHECKSUM_OFFSET)[0]
            if checksum != compute_checksum(frame):
                raise FrameChecksumError('Checksum failed.')

        # Get frame core (counts)s
        d = unpack_from(self.frame_core_format, frame,
                        offset=self.FRAME_HEADER_LEN + 4)  # 4 + 28: registration + header size
        data.c_ref[:] = np.array(d[0::4], dtype=np.uint16)
        data.a_ref[:] = np.array(d[1::4], dtype=np.uint16)
        data.c_sig[:] = np.array(d[2::4], dtype=np.uint16)
        data.a_sig[:] = np.array(d[3::4], dtype=np.uint16)

        return data

    def calibrate_frame(self, frame, get_auxiliaries=False):
        # Apply following processing steps:
        #       convert engineering units (counts) to scientific units (1/m)
        #       Remove clean water offset (from the instrument device file)
        #       Apply instrument linear temperature correction (using constants in instrument device file)

        # Check
        if frame.serial_number != self.serial_number:
            raise SerialNumberIncorrectError('Serial number incorrect.')
        if frame.output_wavelength != self.output_wavelength:
            raise NumberWavelengthIncorrectError('Number of wavelength incorrect.')

        # Process
        internal_temperature_su = compute_internal_temperature(frame.t_int)
        if internal_temperature_su < self.t[0] or self.t[-1] < internal_temperature_su:
            flag_outside_T_cal_range = True
        else:
            flag_outside_T_cal_range = False
        if SCIPY_IMPORTED:
            delta_t_c = self.f_delta_t_c(internal_temperature_su)
            delta_t_a = self.f_delta_t_a(internal_temperature_su)
        else:
            # Use numpy for interpolation (slower as do every wavelength one by one)
            delta_t_c = [np.interp(internal_temperature_su, self.t, v) for v in self.delta_t_c]
            delta_t_a = [np.interp(internal_temperature_su, self.t, v) for v in self.delta_t_a]
        c = (self.offset_c - (1 / self.x) * np.log(frame.c_sig / frame.c_ref)) - delta_t_c
        a = (self.offset_a - (1 / self.x) * np.log(frame.a_sig / frame.a_ref)) - delta_t_a
        if get_auxiliaries:
            external_temperature_su = compute_external_temperature(frame.t_ext)
            return c, a, internal_temperature_su, external_temperature_su, flag_outside_T_cal_range
        else:
            return c, a, flag_outside_T_cal_range

# if __name__ == '__main__':

    # Define constants
    # device_file = '/Users/nils/Data/EXPORTS/DeviceFiles/acs301.dev'
    # device_file = '/Users/nils/Data/EXPORTS/DeviceFiles/acs024.dev'
    # device_file = '/Users/nils/Data/EXPORTS/DeviceFiles/ACS298_20171215/acs298.dev'
    # bin_file = '/Users/nils/Data/EXPORTS/InLine/raw/ACS298/acs298_20180820181008.bin'

    # Test ACS.__init__, ACS.read_device_file
    # acs = ACS(device_file)
    # print(acs)

    # Count frames
    # class CountFrames(BinReader):
    #
    #     COUNTER_GOOD = 0
    #     COUNTER_BAD = 0
    #
    #     def __init__(self, device_filename=None, bin_filename=None):
    #         self.acs = ACS(device_filename)
    #         super(CountFrames, self).__init__(bin_filename)
    #
    #     def handle_frame(self, frame):
    #         try:
    #             self.acs.calibrate_frame(self.acs.unpack_frame(frame))
    #             self.COUNTER_GOOD += 1
    #         except FrameIncompleteError:
    #             self.COUNTER_BAD += 1


    # r = CountFrames(device_file, bin_file)
    # print(r.COUNTER_GOOD, r.COUNTER_BAD)


    # Test convert binnary file to csv file
    # write_auxiliaries = False
    # ConvertBinToCSV(device_file, bin_file, 'out.csv', write_auxiliaries)
    # write_auxiliaries = True
    # ConvertBinToCSV(device_file, bin_file, 'out_with_aux.csv', write_auxiliaries)
