from __future__ import print_function

import numpy as np
from math import log
from collections import namedtuple
from struct import unpack_from, calcsize
from struct import error as struct_error
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


class FrameLengthError(ACSError):
    pass


class OutputWavelengthError(ACSError):
    pass


class FrameTypeError(ACSError):
    pass


class SerialNumberError(ACSError):
    pass


class BinReader:
    READ_SIZE = 1024

    def __init__(self, instrument=None, filename=None):
        self.buffer = bytearray()
        self.instrument = instrument
        if filename:
            self.run(filename)

    def run(self, filename, reset_buffer=True):
        if reset_buffer:
            self.buffer = bytearray()
        with open(filename, 'rb') as f:
            data = f.read(self.READ_SIZE)
            while data:
                self.data_read(data)
                data = f.read(self.READ_SIZE)

    def data_read(self, data):
        self.buffer.extend(data)
        frame = True
        while frame:
            # Get Frame
            frame, valid, self.buffer, unknown_bytes = self.instrument.find_frame(self.buffer)
            if frame and valid:
                self.handle_frame(frame)
            if frame and not valid:
                self.handle_bad_frame(frame)
            if unknown_bytes:
                self.handle_unknown_bytes(unknown_bytes)

    def handle_frame(self, frame, checksum):
        raise NotImplementedError('Implement functionality in handle frame')

    def handle_bad_frame(self, bad_frame):
        pass

    def handle_unknown_bytes(self, bad_bytes):
        pass


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
            fieldnames.extend(['internal_temperature', 'external_temperature'])
        self.f = open(filename, 'w')
        self.writer = csv.writer(self.f)  # , fieldnames=fieldnames)
        # self.writer.writeheader()
        self.writer.writerow(fieldnames)

    def write(self, raw, cal):
        if self.write_auxiliaries:
            self.writer.writerow([raw.time_stamp]
                                 + ["%.6f" % v for v in list(cal.c)]
                                 + ["%.6f" % v for v in list(cal.a)]
                                 + ["%.2f" % cal.internal_temperature]
                                 + ["%.2f" % cal.external_temperature])
        else:
            self.writer.writerow([raw.time_stamp]
                                 + ["%.6f" % v for v in list(cal.c)]
                                 + ["%.6f" % v for v in list(cal.a)])

    def close(self):
        self.f.close()


class ConvertBinToCSV(BinReader):

    def __init__(self, device_filename, bin_filename, csv_filename=None, write_auxiliaries=False):
        if not csv_filename:
            csv_filename = bin_filename + '.dat'
        self.calibrate_auxiliaries = write_auxiliaries
        self.counter_good = 0
        self.counter_bad = 0
        acs = ACS(device_filename)
        self.csv = CSVWriter(acs.lambda_c, acs.lambda_a, write_auxiliaries)
        self.csv.open(csv_filename)
        super(ConvertBinToCSV, self).__init__(acs, bin_filename)

    def handle_frame(self, frame):
        data_raw = self.instrument.unpack_frame(frame)
        try:
            self.instrument.check_data(data_raw)
        except (FrameLengthError, FrameTypeError, SerialNumberError):
            self.counter_bad += 1
            return
        data_cal = self.instrument.calibrate_frame(data_raw, get_external_temperature=self.calibrate_auxiliaries)
        self.counter_good += 1
        self.csv.write(data_raw, data_cal)

    def handle_bad_frame(self, bad_frame):
        self.counter_bad += 1

    def __del__(self):
        self.csv.close()


RawFrameContainer = namedtuple('RawFrameContainer', ['frame_len', 'frame_type', 'serial_number',
                                                     'a_ref_dark', 'p', 'a_sig_dark',
                                                     't_ext', 't_int',
                                                     'c_ref_dark', 'c_sig_dark',
                                                     'time_stamp', 'output_wavelength',
                                                     'c_ref', 'a_ref', 'c_sig', 'a_sig'])
CalibratedFrameContainer = namedtuple('CalibratedFrameContainer',
                                      ['c', 'a', 'internal_temperature', 'external_temperature',
                                       'flag_outside_calibration_range'])


class ACS:
    """
    Unpack and calibrate ACS and AC9 meters attenuation (c) and absorption (a) engineering values
    (sig, ref counts in binary) to scientific units (1/m).
    """

    REGISTRATION_BYTES = b'\xff\x00\xff\x00'
    REGISTRATION_BYTES_LENGTH = len(REGISTRATION_BYTES)
    FRAME_HEADER_DESCRIPTOR = '!HBBlHHHHHHHIBB'
    FRAME_HEADER_LENGTH = calcsize(FRAME_HEADER_DESCRIPTOR)

    def __init__(self, device_filename=None):
        # Meta data
        self.serial_number = None  # and Meter Type
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
        self.frame_length = None
        self.frame_descriptor = None

        if device_filename:
            self.read_device_file(device_filename)

    def __repr__(self):
        """
        Representation of instrument. Most values comes from device file.
        :return: string
        """
        return 'SN: ' + self.serial_number + ' ' + \
               self.get_meter_type_str() + ' ' + self.get_serial_number_str() + '\n' + \
               'V: ' + str(self.structure_version_number) + '\n' + \
               'BD: ' + str(self.baudrate) + '\n' + \
               'WL: ' + str(self.output_wavelength) + '\n' + \
               'WL(c): ' + str(self.lambda_c.size) + ' ' + str(self.lambda_c) + '\n' + \
               'WL(a): ' + str(self.lambda_a.size) + ' ' + str(self.lambda_a) + '\n' + \
               'T:' + str(self.t.size) + ' ' + str(self.t) + '\n' + \
               'DT(c): ' + str(np.size(self.delta_t_c, 0)) + ' ' + str(np.size(self.delta_t_c, 1)) + ' ' \
               + str(self.delta_t_c) + '\n' + \
               'DT(a): ' + str(np.size(self.delta_t_a, 0)) + ' ' + str(np.size(self.delta_t_a, 1)) + ' ' \
               + str(self.delta_t_a) + '\n' + \
               'WO(c): ' + str(self.offset_c.size) + ' ' + str(self.offset_c) + '\n' + \
               'WO(a): ' + str(self.offset_a.size) + ' ' + str(self.offset_a) + '\n' + \
               'FMT_F: ' + str(self.frame_descriptor) + '\n'

    def get_meter_type_str(self):
        """
        Convert meter type from hexadecimal to string
        :return: instrument type (string)
        """
        if self.serial_number:
            if self.serial_number[:4] == '0x53':
                return 'ACS'
            else:
                return 'UnknownMeterType'
        else:
            return ''

    def get_serial_number_str(self):
        """
        Convert serial number from hexadecimal to string
        :return: instrument serial number (string)
        """
        if self.serial_number:
            return str(int(self.serial_number[-6:], 16))
        else:
            return ''

    def read_device_file(self, filename):
        """
        Parse instrument device file. Required to be able to unpack and calibrate a frame.
        :param filename: path to ACS or AC9 device file
        :return:
        """
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
                    self.output_wavelength = int(l.split(';')[0])
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
                # Use scipy for interpolation (build 2D interpolation function, faster than numpy)
                self.f_delta_t_c = interpolate.interp1d(self.t, self.delta_t_c, axis=1, assume_sorted=True, copy=False,
                                                        bounds_error=False,
                                                        fill_value=(self.delta_t_c[:, 1], self.delta_t_c[:, -1]))
                self.f_delta_t_a = interpolate.interp1d(self.t, self.delta_t_a, axis=1, assume_sorted=True, copy=False,
                                                        bounds_error=False,
                                                        fill_value=(self.delta_t_a[:, 1], self.delta_t_a[:, -1]))
            else:
                # Use numpy for interpolation (slower as do every wavelength one by one)
                self.f_delta_t_c = lambda tq: [np.interp(tq, self.t, v) for v in self.delta_t_c]
                self.f_delta_t_a = lambda tq: [np.interp(tq, self.t, v) for v in self.delta_t_a]
            self.set_frame_descriptor()

    def set_frame_descriptor(self):
        """
        Set frame format description to unpack frame from C-struct to python variables
        Make string descriptor to unpack the binary frame
          The frame format and length is a function of the number of wavelength
        # Use network format (same as big-endian but mention that it will go over network)
        fmt = '!'
        # 2 bytes: packet length
        fmt += 'H'
        # 1 byte: Packet type identifier
        fmt += 'B'
        # 1 byte: reserved for future use
        fmt += 'B'
        # 4 bytes long integer: Meter Type + Instrument Serial Number
        fmt += 'L'
        # 2 bytes: A reference dark counts (for diagnostic purpose)
        fmt += 'H'
        # 2 bytes: A/D counts from the pressure sensor circuitry
        fmt += 'H'
        # 2 bytes: A signal dark counts (for diagnostic purpose)
        fmt += 'H'
        # 2 bytes: External temperature voltage counts
        fmt += 'H'
        # 2 bytes unsigned integer:  Internal temperature voltage counts
        fmt += 'H'
        # 2 bytes: C reference dark counts
        fmt += 'H'
        # 2 bytes: C signal dark counts
        fmt += 'H'
        # 4 bytes unsigned integer: Time stamp (ms)
        fmt += 'I'
        # 1 byte: reserved for future use
        fmt += 'B'
        # 1 byte unsigned integer: Number of output wavelength
        fmt += 'B'
        2 bytes unsigned: Data for scan (c_ref, a_ref, c_sig, a_sig ... )
        for i in range(output_wavelength):
            self.frame_core_format += 'HHHH'
        # 2 bytes: Check sum (Not included in frame)
        fmt += 'H'
        # 1 byte: Last character 0x00 (Not included in frame, not always 0x00)
        fmt += 'c'

        :return:
        """
        self.frame_descriptor = self.FRAME_HEADER_DESCRIPTOR
        for i in range(self.output_wavelength):
            self.frame_descriptor += 'HHHH'
        self.frame_length = self.REGISTRATION_BYTES_LENGTH + calcsize(self.frame_descriptor)

    def find_frame(self, buffer):
        """
        Find the first and complete frame from the buffer
        :param buffer: byte array
        :return: frame: first frame found
                 checksum: boolean indicating if valid or invalid frame
                 buffer_post_frame: buffer left after the frame
                 buffer_pre_frame: buffer preceding the first frame returned (likely unknown frame header)
        """
        try:
            # Look for registration bytes
            i = buffer.index(self.REGISTRATION_BYTES)
            # Take care of special case when checksum + pad byte or just checksum = \xff\x00
            # It's unlikely that the full packet length is equal to \xff\x00 = 65280
            while buffer.find(self.REGISTRATION_BYTES, i + 2, i + 2 + self.REGISTRATION_BYTES_LENGTH) != -1:
                i += 2
            # Get Length of frame (following 2 bytes, already know it from device file)
            # frame_length = unpack_from('!H', buffer, offset=i + self.REGISTRATION_BYTES_LENGTH)
            frame_end_index = i + self.frame_length
            # Get frame
            frame = buffer[i:frame_end_index]
            if len(frame) != self.frame_length:
                return bytearray(), None, buffer, bytearray()
            # Get Checksum
            checksum = buffer[frame_end_index:frame_end_index + 2]
            if len(checksum) != 2:
                return bytearray(), None, buffer, bytearray()
            # Check checksum
            if not self.valid_frame(frame, checksum):
                # Error in frame, remove registration bytes and attempt again
                return frame, False, buffer[i+self.REGISTRATION_BYTES_LENGTH:],\
                       buffer[:i+self.REGISTRATION_BYTES_LENGTH]
            # Pad byte is not always present... (only +2 for checksum)
            return frame, True, buffer[frame_end_index + 2:], buffer[:i]
        except ValueError:
            # No registration byte found
            return bytearray(), None, buffer, bytearray()
        except struct_error:
            # Buffer is too short to unpack packet length
            return bytearray(), None, buffer, bytearray()

    @staticmethod
    def valid_frame(frame, checksum_received):
        """
        Compute frame checksum and compare it to received checksum
            The checksum is the unsigned 16 bit sum of all bytes received in packet,
            including the registration bytes, up to the last byte preceding the checksum bytes.
        :param frame: frame including registration bytes
        :param checksum_received: two bytes checksum received with the frame
        :return: True: checksum received and computed match
                 False: checksum received and computed do not match
        """
        # Frame length could be check but is not as checksum should fail if frame length is incorrect
        return np.uint16(sum(frame)) == unpack_from('!H', checksum_received)

    def unpack_frame(self, frame):
        """
        Convert frame from C structs to Python values
        Assume valid frame
        :param frame: byte array including registration bytes
        :return: data: a frame container tuple with python values from the frame
        """
        d = unpack_from(self.frame_descriptor, frame, offset=self.REGISTRATION_BYTES_LENGTH)
        return RawFrameContainer(frame_len=d[0],  # packet length
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
                                 output_wavelength=d[13],  # number of output wavelength
                                 c_ref=np.array(d[14::4], dtype=np.uint16), a_ref=np.array(d[15::4], dtype=np.uint16),
                                 c_sig=np.array(d[16::4], dtype=np.uint16), a_sig=np.array(d[17::4], dtype=np.uint16))

    def check_data(self, data):
        """
        Raise an exception if anything is unexpected with the data
        :param data: a RawFrameContainer with python values from the frame (typically obtained from unpack_frame)
        :return: True if passed all tests otherwise raise error
        """
        # if self.frame_length != data.frame_len:  # Not needed as passed checksum
        #     raise FrameLengthError('Frame length not matching descriptor from device file.')
        if data.frame_type < 3:  # 3 or higher for AC-S
            raise FrameTypeError('Frame type incorrect (not AC-S).')
        if data.serial_number != self.serial_number:
            raise SerialNumberError('Serial number incorrect.')
        # if data.output_wavelength != self.output_wavelength:  # Not needed as tested with frame_length
        #     raise OutputWavelengthError('Number of wavelength not matching descriptor from device file.')
        return True

    @staticmethod
    def compute_external_temperature(counts):
        """
        Convert external temperature engineering units (counts) to scientific units (deg C)
        :param counts: temperature in engineering units (counts)
        :return: temperature in scientific units (deg C)
        """
        return -7.1023317e-13 * counts ** 3 + \
               7.09341920e-8 * counts ** 2 + \
               -3.87065673e-3 * counts + 95.8241397

    @staticmethod
    def compute_internal_temperature(counts):
        """
        Convert internal temperature engineering units (counts) to scientific units (deg C)
        :param counts: temperature in engineering units (counts)
        :return: temperature in scientific units (deg C)
        """
        volts = 5 * counts / 65535
        resistance = 10000 * volts / (4.516 - volts)
        return 1 / (0.00093135 + 0.000221631 * log(resistance) + 0.000000125741 * log(resistance) ** 3) - 273.15

    def calibrate_frame(self, frame, get_external_temperature=False):
        """
        Calibrate frame (assumed valid) by following these steps:
               + convert engineering units (counts) to scientific units (1/m)
               + remove clean water offset (from the instrument device file)
               + apply instrument linear temperature correction (using constants in instrument device file)
        :param frame: a named tuple with python values from the frame (typically obtained from unpack_frame)
        :param get_external_temperature: compute external temperature from sensor
        :return: CalibratedFrameContainer with or without external temperature
        """
        # Compute internal temperatures
        internal_temperature_su = self.compute_internal_temperature(frame.t_int)
        if internal_temperature_su < self.t[0] or self.t[-1] < internal_temperature_su:
            flag_outside_calibration_range = True
        else:
            flag_outside_calibration_range = False
        # Interpolate temperature from correction tables
        delta_t_c = self.f_delta_t_c(internal_temperature_su)
        delta_t_a = self.f_delta_t_a(internal_temperature_su)
        # Calibrate and apply temperature and clean water offset corrections
        c = (self.offset_c - (1 / self.x) * np.log(frame.c_sig / frame.c_ref)) - delta_t_c
        a = (self.offset_a - (1 / self.x) * np.log(frame.a_sig / frame.a_ref)) - delta_t_a
        # Pack output in named tuple
        if get_external_temperature:
            return CalibratedFrameContainer(c=c, a=a,
                                            internal_temperature=internal_temperature_su,
                                            external_temperature=self.compute_external_temperature(frame.t_ext),
                                            flag_outside_calibration_range=flag_outside_calibration_range)
        else:
            return CalibratedFrameContainer(c=c, a=a,
                                            internal_temperature=internal_temperature_su,
                                            external_temperature=None,
                                            flag_outside_calibration_range=flag_outside_calibration_range)
