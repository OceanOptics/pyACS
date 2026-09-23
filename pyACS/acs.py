from __future__ import print_function

import os
from datetime import datetime, timezone
import numpy as np
from math import log
from collections import namedtuple
from struct import unpack_from, calcsize
import csv
from sys import version_info, exit
try:
    from scipy import interpolate
except ImportError:
    interpolate = None

try:
    import pandas as pd
except ImportError:
    pd = None

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

    def handle_frame(self, frame):
        raise NotImplementedError('Implement functionality in handle frame')

    def handle_bad_frame(self, bad_frame):
        pass

    def handle_unknown_bytes(self, bad_bytes):
        pass


class CSVWriter:

    def __init__(self, filename=None, lambda_c=None, lambda_a=None, write_auxiliaries=False):
        self._lambda_c = lambda_c
        self._lambda_a = lambda_a
        self._write_auxiliaries = write_auxiliaries

        self._filename = filename
        self._f = None
        self._writer = None

    def open(self, filename=None, write_external_timestamp=False):
        if self._f is not None:
            self.close()
        if filename is not None:
            self._filename = filename
        if self._filename is None:
            raise ValueError('CSVWriter: no filename provided to open output file.')

        fieldnames = ['external_timestamp'] if write_external_timestamp else []
        fieldnames.extend(['timestamp'] + ['c%3.1f' % x for x in self._lambda_c] + ['a%3.1f' % x for x in self._lambda_a])
        if self._write_auxiliaries:
            fieldnames.extend(['internal_temperature', 'external_temperature'])
        fieldnames.append('flag_outside_calibration_range')

        self._f = open(self._filename, 'w')
        self._writer = csv.writer(self._f)
        self._writer.writerow(fieldnames)

    def write(self, raw, cal):
        has_external_timestamp = raw.external_timestamp is not None
        if self._f is None or self._writer is None:
            self.open(write_external_timestamp=has_external_timestamp)
        row = []
        if has_external_timestamp:
            if np.isfinite(raw.external_timestamp):
                row = [datetime.fromtimestamp(raw.external_timestamp, tz=timezone.utc).replace(tzinfo=None).isoformat()]
            else:
                row = ['NaT']
        row.extend([raw.timestamp] + ["%.6f" % v for v in list(cal.c)] + ["%.6f" % v for v in list(cal.a)])
        if self._write_auxiliaries:
            row.extend(["%.2f" % cal.internal_temperature] + ["%.2f" % cal.external_temperature])
        row.append(cal.flag_outside_calibration_range)
        self._writer.writerow(row)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class ConvertBinToCSV(BinReader):

    def __init__(self, acs, bin_filename, csv_filename=None, write_auxiliaries=True):
        if not csv_filename:
            csv_filename = bin_filename + '.dat'
        self.calibrate_auxiliaries = write_auxiliaries
        self.counter_good = 0
        self.counter_bad = 0
        self.csv = CSVWriter(csv_filename, acs.lambda_c, acs.lambda_a, write_auxiliaries)
        try:
            super(ConvertBinToCSV, self).__init__(acs, bin_filename)
        finally:
            self.csv.close()  # ensure file is flushed/closed

    def handle_frame(self, frame, timestamp=None):
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


class BinToDataFrame(BinReader):

    ANC_VAR_NAMES = ['internal_temperature', 'external_temperature', 'flag_outside_calibration_range',
                     'a_ref_dark', 'a_sig_dark', 'c_ref_dark', 'c_sig_dark']

    def __init__(self, *args, **kwargs):
        # Parsed data
        self.timestamp = None
        self.c = None
        self.a = None
        self.int_temp = None
        self.ext_temp = None
        self.flag_outside_calibration_range = None
        self.a_ref_dark = None
        self.a_sig_dark = None
        self.c_ref_dark = None
        self.c_sig_dark = None
        # Index
        self.index = 0
        self.c_labels = None
        self.a_labels = None
        # External timestamp
        self.external_timestamp = None

        super(BinToDataFrame, self).__init__(*args, **kwargs)

    def init_arrays(self, filename):
        # Estimate byte number (can't be underestimated)
        n = round(os.path.getsize(filename) / (self.instrument.frame_length))
        self.timestamp = np.empty(n, dtype=np.int64)
        self.c = np.empty([n, self.instrument.output_wavelength], dtype=np.float64)
        self.a = np.empty([n, self.instrument.output_wavelength], dtype=np.float64)
        self.int_temp = np.empty(n, dtype=np.float64)
        self.ext_temp = np.empty(n, dtype=np.float64)
        self.flag_outside_calibration_range = np.empty(n, dtype=np.bool)
        self.a_ref_dark = np.empty(n, dtype=np.int64)
        self.a_sig_dark = np.empty(n, dtype=np.int64)
        self.c_ref_dark = np.empty(n, dtype=np.int64)
        self.c_sig_dark = np.empty(n, dtype=np.int64)
        self.external_timestamp = np.empty(n, dtype=np.float64)
        self.index = 0
        self.c_labels = ['c%3.1f' % x for x in self.instrument.lambda_c]
        self.a_labels = ['a%3.1f' % x for x in self.instrument.lambda_a]

    def clean_arrays(self):
        # Keep only data points within index (other are from initialization and not used)
        self.timestamp = self.timestamp[:self.index]
        self.c = self.c[:self.index, :]
        self.a = self.a[:self.index, :]
        self.int_temp = self.int_temp[:self.index]
        self.ext_temp = self.ext_temp[:self.index]
        self.flag_outside_calibration_range = self.flag_outside_calibration_range[:self.index]
        self.a_ref_dark = self.a_ref_dark[:self.index]
        self.a_sig_dark = self.a_sig_dark[:self.index]
        self.c_ref_dark = self.c_ref_dark[:self.index]
        self.c_sig_dark = self.c_sig_dark[:self.index]
        self.external_timestamp = self.external_timestamp[:self.index]

    def pack_data_frame(self):
        if pd is None:
            raise ImportError('pandas is required to pack data frame')
        if len(self.external_timestamp) and not np.all(np.isnan(self.external_timestamp)):
            return pd.DataFrame(zip(*[
                pd.to_datetime(self.external_timestamp, unit='s'), #, utc=True),
                self.timestamp,
                *[v for v in self.c.transpose()],
                *[v for v in self.a.transpose()],
                self.int_temp, self.ext_temp,
                self.flag_outside_calibration_range,
                self.a_ref_dark, self.a_sig_dark,
                self.c_ref_dark, self.c_sig_dark
            ]), columns=['external_timestamp', 'timestamp', *self.c_labels, *self.a_labels, *self.ANC_VAR_NAMES])
        return pd.DataFrame(zip(*[self.timestamp,
                                  *[v for v in self.c.transpose()],
                                  *[v for v in self.a.transpose()],
                                  self.int_temp, self.ext_temp,
                                  self.flag_outside_calibration_range,
                                  self.a_ref_dark, self.a_sig_dark,
                                  self.c_ref_dark, self.c_sig_dark]),
                            columns=['timestamp', *self.c_labels, *self.a_labels, *self.ANC_VAR_NAMES])

    def run(self, filename, *args, **kwargs):
        self.init_arrays(filename)
        super(BinToDataFrame, self).run(filename, *args, **kwargs)
        self.clean_arrays()
        return self.pack_data_frame()

    def handle_frame(self, frame):
        data_raw = self.instrument.unpack_frame(frame)
        try:
            self.instrument.check_data(data_raw)
        except (FrameLengthError, FrameTypeError, SerialNumberError):
            print('Check data failed')
            return
        data_cal = self.instrument.calibrate_frame(data_raw, get_external_temperature=True)

        self.timestamp[self.index] = data_raw.timestamp
        self.a_ref_dark[self.index] = data_raw.a_ref_dark
        self.a_sig_dark[self.index] = data_raw.a_sig_dark
        self.c_ref_dark[self.index] = data_raw.c_ref_dark
        self.c_sig_dark[self.index] = data_raw.c_sig_dark
        self.c[self.index, :] = data_cal.c
        self.a[self.index, :] = data_cal.a
        self.int_temp[self.index] = data_cal.internal_temperature
        self.ext_temp[self.index] = data_cal.external_temperature
        self.flag_outside_calibration_range[self.index] = data_cal.flag_outside_calibration_range
        self.external_timestamp[self.index] = data_raw.external_timestamp
        self.index += 1

    # def handle_bad_frame(self, bad_frame):
    #     print('Checksum failed after frame %d' % self.index)
    #     print(bad_frame)

    # def handle_unknown_bytes(self, bdata):
    #     print(bdata)


RawFrameContainer = namedtuple('RawFrameContainer', ['frame_len', 'frame_type', 'serial_number',
                                                     'a_ref_dark', 'p', 'a_sig_dark',
                                                     't_ext', 't_int',
                                                     'c_ref_dark', 'c_sig_dark',
                                                     'timestamp', 'output_wavelength',
                                                     'c_ref', 'a_ref', 'c_sig', 'a_sig',
                                                     'checksum', 'external_timestamp'])
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
    CHECKSUM_FORMAT = 'H'
    CHECKSUM_LENGTH = calcsize('!' + CHECKSUM_FORMAT)
    PAD_BYTE_FORMAT = 'c'
    PAD_BYTE_LENGTH = calcsize('!' + PAD_BYTE_FORMAT)
    EXTERNAL_TIMESTAMP_FORMAT = '' # 'd'
    EXTERNAL_TIMESTAMP_LENGTH = 0  # calcsize('!' + EXTERNAL_TIMESTAMP_FORMAT)

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
                    self.t = np.array(l.split(';')[0].strip().split('\t')).astype(float)
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
            if interpolate is not None:
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
        if self.CHECKSUM_FORMAT:
            self.frame_descriptor += self.CHECKSUM_FORMAT
        if self.PAD_BYTE_FORMAT:
            self.frame_descriptor += self.PAD_BYTE_FORMAT
        if self.EXTERNAL_TIMESTAMP_FORMAT:
            self.frame_descriptor += self.EXTERNAL_TIMESTAMP_FORMAT
        self.frame_length = self.REGISTRATION_BYTES_LENGTH + calcsize(self.frame_descriptor)
        self.checksum_unpacked_index = -1 - 1 * bool(self.PAD_BYTE_LENGTH) - 1 * bool(self.EXTERNAL_TIMESTAMP_LENGTH)

    def find_frame(self, buffer):
        """
        Find the first and complete frame from the buffer
        :param buffer: byte array
        :return: frame: first frame found
                 checksum: boolean indicating if valid or invalid frame
                 buffer_post_frame: buffer left after the frame
                 buffer_pre_frame: buffer preceding the first frame returned (likely unknown frame header)
        """
        frame, is_valid, buffer_post_frame, buffer_pre_frame = bytearray(), None, buffer, bytearray()
        # Look for registration bytes
        i = buffer.find(self.REGISTRATION_BYTES)
        if i == -1:
            # No registration byte found
            return bytearray(), None, buffer, bytearray()
        # Take care of edge case: end of previous corrupted frame (last byte of checksum + pad byte) = \xff\x00
        while buffer.find(self.REGISTRATION_BYTES, i + 2, i + 2 + self.REGISTRATION_BYTES_LENGTH) != -1:
            i += 2
        # Assume pad_byte is always present (even if not 0x00), old comments referred to instance with missing pad_byte that would throw off the frame length and datetime.
        frame_end_index = i + self.frame_length  # Includes checksum, pad_byte, and external timestamp if enabled
        # Make sure buffer is long enough
        if len(buffer) < frame_end_index:
            return bytearray(), None, buffer, bytearray()
        # Get frame and checksum
        frame = buffer[i:frame_end_index]
        # Checksum validity
        if self.valid_frame(frame) == False:  # Needed to distinguish None (no checksum) and False (invalid checksum)
            # Error in frame, remove registration bytes and attempt again
            return frame, False, buffer[i+self.REGISTRATION_BYTES_LENGTH:], buffer[:i+self.REGISTRATION_BYTES_LENGTH]
        # Return complete frame (with checksum, pad byte, and external timestamp if enabled)
        return frame, True, buffer[frame_end_index:], buffer[:i]

    def valid_frame(cls, frame):
        """
        If checksum available then use chesksum to validate frame
        otherwise check length if no other REGISTRATION_BYTES found in frame and timestamp is valid (if enabled)
        Compute frame checksum and compare it to received checksum
            The checksum is the unsigned 16 bit sum of all bytes received in packet,
            including the registration bytes, up to the last byte preceding the checksum bytes.
        :param frame: frame including registration bytes
        :return: None: no cheksum to check (checksum length = 0)
                 True: checksum received and computed match
                 False: checksum received and computed do not match
        """
        if cls.CHECKSUM_LENGTH:
            end_index_offset = cls.CHECKSUM_LENGTH + cls.PAD_BYTE_LENGTH + cls.EXTERNAL_TIMESTAMP_LENGTH
            checksum_computed = np.uint16(sum(frame[:-end_index_offset]) % 2 ** 16)
            checksum_received = unpack_from('!H', frame[-end_index_offset:-end_index_offset+cls.CHECKSUM_LENGTH])[0]
            return checksum_computed == checksum_received
        if (frame[:cls.REGISTRATION_BYTES_LENGTH] != cls.REGISTRATION_BYTES or
                frame.find(cls.REGISTRATION_BYTES, cls.REGISTRATION_BYTES_LENGTH) != -1):
            # Frame does not start with registration bytes or has additional registration bytes in the middle of the frame
            # Not checking for length as find_frame gives frame of accurate length, and unpack would raise an error
            return False
        if cls.EXTERNAL_TIMESTAMP_LENGTH:
            try:
                t = unpack_from('!d', frame[-cls.EXTERNAL_TIMESTAMP_LENGTH:])[0]
                if np.isfinite(t):
                    datetime.fromtimestamp(t, tz=timezone.utc)
            except OverflowError:
                return False
            return True
        return None

    def unpack_frame(self, frame):
        """
        Convert frame from C structs to Python values
        Assume valid frame
        :param frame: byte array including registration bytes
        :return: data: a frame container tuple with python values from the frame
        """
        d = unpack_from(self.frame_descriptor, frame, offset=self.REGISTRATION_BYTES_LENGTH)
        n = 4 * self.output_wavelength
        if d[11] == 20168464:
            print(d)
            print(frame)
            print(frame.hex())
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
                                 timestamp=d[11],  # unsigned integer: Instrument Timestamp (ms)
                                 # data[] = d[12] # reserved for future use
                                 output_wavelength=d[13],  # number of output wavelength
                                 c_ref=np.array(d[14:14+n:4], dtype=np.uint16),
                                 a_ref=np.array(d[15:15+n:4], dtype=np.uint16),
                                 c_sig=np.array(d[16:16+n:4], dtype=np.uint16),
                                 a_sig=np.array(d[17:17+n:4], dtype=np.uint16),
                                 checksum=d[self.checksum_unpacked_index] if self.CHECKSUM_LENGTH > 0 else None,
                                 # pad_byte=d[-2] if self.PAD_BYTE_LENGTH > 0 else None,
                                 external_timestamp=d[-1] if self.EXTERNAL_TIMESTAMP_LENGTH > 0 else None)

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
        with np.errstate(divide='ignore', invalid='ignore'):  # c_sig, c_ref, a_sig, and a_ref can be zero
            c = (self.offset_c - (1 / self.x) * np.log(frame.c_sig / frame.c_ref)) - delta_t_c
            a = (self.offset_a - (1 / self.x) * np.log(frame.a_sig / frame.a_ref)) - delta_t_a
        # Pack output in named tuple
        return CalibratedFrameContainer(
            c=c, a=a,
            internal_temperature=internal_temperature_su,
            external_temperature=self.compute_external_temperature(frame.t_ext) if get_external_temperature is not None else None,
            flag_outside_calibration_range=flag_outside_calibration_range
        )


class ACSCompass(ACS):
    pass


class ACSInlinino(ACS):

    CHECKSUM_FORMAT = '' #'H'
    CHECKSUM_LENGTH = 0  # calcsize('!' + CHECKSUM_FORMAT)
    PAD_BYTE_FORMAT = '' # 'c'
    PAD_BYTE_LENGTH = 0  # calcsize('!' + PAD_BYTE_FORMAT)
    EXTERNAL_TIMESTAMP_FORMAT = 'd'
    EXTERNAL_TIMESTAMP_LENGTH = calcsize('!' + EXTERNAL_TIMESTAMP_FORMAT)

