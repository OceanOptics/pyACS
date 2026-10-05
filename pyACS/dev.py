from datetime import datetime
import numpy as np
import os
import re
from struct import calcsize
from scipy import interpolate
import xarray as xr


class Dev():

    """
    A class for parsing ACS calibration (.dev) files. Retrieves most metadata from the file.
    Dev files are necessary for converting binary data to something more meaningful.

    This class can be instantiated and data can be accessed as class attributes.

    Example Usage:
        dev = Dev('/home/jovyan/cals/acs011.dev')
        acs_sn = dev.sn

    Alternatively, the to_ds() function will format calibration data so that it can be used with
    operations based in xarray. The to_nc() function will export calibration data as a netcdf.
    """

    def __init__(self, filepath: os.path.abspath) -> None:
        """
        Parse the .dev file. Functions in this __init__ seek out specific metadata.

        :param filepath: The location of the dev file.
        """
        self.filepath = os.path.normpath(filepath)
        self.__read_dev()
        self.__parse_metadata()
        self.__parse_tbins()
        self.__parse_offsets()
        self.__check_parse()
        self.__build_frame_descriptor()


    def __read_dev(self) -> None:
        """Import the .dev file as a text file."""

        with open(self.filepath, 'r') as _file:
            self._lines = _file.readlines()


    def __parse_metadata(self) -> None:
        """
        Parse the .dev file for sensor metadata.
        The end result is file metadata assigned as class attributes.
        """

        metadata_lines = [line for line in self._lines if 'C and A offset' not in line]
        for line in metadata_lines:
            if 'ACS Meter' in line:
                self.sensor_type = re.findall('(.*?)\n', line)[0]
            elif 'Serial' in line:
                self.sn_hexdec = re.findall('(.*?)\t', line)[0]
                self.sn = self.serial_number = 'ACS' + str(int(self.sn_hexdec[-6:], 16)).zfill(3)
            elif 'structure version' in line:
                self.structure_version = self.structure_version_number = int(re.findall('(.*?)\t', line)[0])
            elif 'tcal' in line:
                self.tcal, self.ical = [float(v) for v in re.findall(': [+-]?([0-9]*[.]?[0-9]+) C', line)]
                cal_date_str = re.findall('file on (.*?)[.].*?\n', line)[0].replace(' ', '')
                try:
                    self.cal_date = datetime.strptime(cal_date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
                except:
                    self.cal_date = datetime.strptime(cal_date_str, '%m/%d/%y').strftime('%Y-%m-%d')
            elif 'Depth calibration' in line:
                (self.depth_cal_offset,
                 self.depth_cal_scale_factor) = [float(v) for v in re.findall('[+-]?([0-9]*[.]?[0-9]+)\t', line)]
            elif 'Baud' in line:
                self.baudrate = int(re.findall('(.*?)\t', line)[0])
            elif 'Path' in line:
                self.path_length = self.x = float(re.findall('(.*?)\t', line)[0])
            elif 'wavelengths' in line:
                self.output_wavelength = int(re.findall('(.*?)\t', line)[0])
            elif 'number of temperature bins' in line:
                self.num_tbins = int(re.findall('(.*?)\t', line)[0])
            elif 'maxANoise' in line:
                (self.max_a_noise, self.max_c_noise, self.max_a_nonconform, self.max_c_nonconform,
                 self.max_a_difference, self.max_c_difference, self.min_a_counts,
                 self.min_c_counts, self.min_r_counts, self.max_tempsdev,
                 self.max_depth_sdev) = [float(v) for v in re.findall('[+-]?([0-9]*[.]?[0-9]+)\t', line)]


    def __parse_tbins(self) -> None:
        """
        Parse the .dev file for temperature bin information.
        The end result is a class attributes that contains an array of temperature bins.
        """

        line = [line for line in self._lines if '; temperature bins' in line][0]
        tbins = line.split('\t')
        tbins = [v for v in tbins if v]
        tbins = [v for v in tbins if v != '\n']
        tbins = [float(v) for v in tbins if 'temperature bins' not in v]
        self.tbins = self.t = np.array(tbins)


    def __parse_offsets(self) -> None:
        """
        Parse the .dev file for a and c offsets.
        The end result is a set of class attributes that contain wavelength offset and deltas as arrays.
        """

        offset_lines = [line for line in self._lines if 'C and A offset' in line]
        c_wavelengths = []
        a_wavelengths = []
        c_offsets = []
        a_offsets = []
        c_deltas = []
        a_deltas = []
        for line in offset_lines:
            offsets, c_delta, a_delta = line.split('\t\t')[:-1]
            wavelength_c, wavelength_a, _, offset_c, offset_a = offsets.split('\t')
            c_wavelengths.append(float(wavelength_c.replace('C', '')))
            a_wavelengths.append(float(wavelength_a.replace('A', '')))
            c_offsets.append(float(offset_c))
            a_offsets.append(float(offset_a))
            c_deltas.append(np.array([float(v) for v in c_delta.split('\t')]))
            a_deltas.append(np.array([float(v) for v in a_delta.split('\t')]))
        self.lambda_c = np.array(c_wavelengths)
        self.lambda_a = np.array(a_wavelengths)
        self.offset_c = np.array(c_offsets)
        self.offset_a = np.array(a_offsets)
        self.delta_t_c = np.array(c_deltas)
        self.delta_t_a = np.array(a_deltas)
        self.f_delta_t_c = interpolate.interp1d(self.tbins, self.delta_t_c, axis=1, assume_sorted=True, copy=False,
                                                bounds_error=False,
                                                fill_value=(self.delta_t_c[:, 1], self.delta_t_c[:, -1]))
        self.f_delta_t_a = interpolate.interp1d(self.tbins, self.delta_t_a, axis=1, assume_sorted=True, copy=False,
                                                bounds_error=False,
                                                fill_value=(self.delta_t_a[:, 1], self.delta_t_a[:, -1]))


    def __build_frame_descriptor(self) -> None:
        """
        Build a frame descriptor for parsing binary ACS packets.
        Only used when reading raw binary from a file or over serial.
        """

        self.REGISTRATION_BYTES = b'\xff\x00\xff\x00'
        self.REGISTRATION_BYTES_LENGTH = len(self.REGISTRATION_BYTES)
        self.FRAME_HEADER_DESCRIPTOR = '!HBBlHHHHHHHIBB'
        self.FRAME_HEADER_LENGTH = calcsize(self.FRAME_HEADER_DESCRIPTOR)

        self.frame_descriptor = self.FRAME_HEADER_DESCRIPTOR
        for i in range(self.output_wavelength):
            self.frame_descriptor += 'HHHH'
        self.frame_length = self.REGISTRATION_BYTES_LENGTH + calcsize(self.frame_descriptor)


    def __check_parse(self) -> None:
        """Verify that the parse obtained the correct information."""

        if len(self.lambda_c) != len(self.lambda_a):
            raise ValueError('Mismatch between number of wavelengths extracted for A and C.')
        if self.delta_t_c.shape != (len(self.lambda_c), self.num_tbins):
            raise ValueError('Mismatch between length of C wavelengths and number of temperature bins.')
        if self.delta_t_a.shape != (len(self.lambda_a), self.num_tbins):
            raise ValueError('Mismatch between length of A wavelengths and number of temperature bins.')


    def to_ds(self) -> xr.Dataset:
        """
        Export class attributes as an xr.Dataset.

        :return: An xarray dataset containing calibration information.
        """

        ds = xr.Dataset()
        ds = ds.assign_coords({'lambda_a': self.lambda_a})
        ds['lambda_a'].attrs['units'] = 'nanometers'
        ds['lambda_a'].attrs['units_tex'] = r'$nm$'
        ds['lambda_a'].attrs['description'] = 'ACS absorption wavelength bins.'

        ds = ds.assign_coords({'lambda_c': self.lambda_c})
        ds['lambda_c'].attrs['units'] = 'nanometers'
        ds['lambda_c'].attrs['units_tex'] = r'$nm$'
        ds['lambda_c'].attrs['description'] = 'ACS attenuation wavelength bins.'

        ds = ds.assign_coords({'temperature_bins': self.tbins})
        ds['temperature_bins'].attrs['units'] = 'degrees_celsius'
        ds['temperature_bins'].attrs['units_tex'] = r'$^{\circ}C$'
        ds['temperature_bins'].attrs['description'] = 'ACS calibration temperature bins.'

        ds['offsets_a'] = (['lambda_a'], np.array(self.offset_a))
        ds['offsets_a'].attrs['units'] = 'inverse_meters'
        ds['offsets_a'].attrs['units_tex'] = r'$\frac{1}{m}$'
        ds['offsets_a'].attrs['description'] = 'Instrument specific absorption offsets.'

        ds['delta_t_a'] = (['lambda_a', 'temperature_bins'], np.array(self.delta_t_a))
        ds['delta_t_a'].attrs['units'] = 'unitless'
        ds['delta_t_a'].attrs['units_tex'] = r''
        ds['delta_t_a'].attrs['description'] = 'Temperature correction deltas for absorption.'

        ds['offsets_c'] = (['lambda_c'], np.array(self.offset_c))
        ds['offsets_c'].attrs['units'] = 'inverse_meters'
        ds['offsets_c'].attrs['units_tex'] = r'$\frac{1}{m}$'
        ds['offsets_c'].attrs['description'] = 'Instrument specific attenuation offsets.'

        ds['delta_t_c'] = (['lambda_c', 'temperature_bins'], np.array(self.delta_t_c))
        ds['delta_t_c'].attrs['units'] = 'unitless'
        ds['delta_t_c'].attrs['units_tex'] = r''
        ds['delta_t_c'].attrs['description'] = 'Temperature correction deltas for attenuation.'

        ds.attrs['sensor_type'] = self.sensor_type
        ds.attrs['serial_number'] = self.sn
        ds.attrs['factory_calibration_date'] = self.cal_date
        ds.attrs['output_wavelengths'] = self.output_wavelength
        ds.attrs['number_temp_bins'] = self.num_tbins
        ds.attrs['path_length'] = self.path_length
        ds.attrs['tcal'] = self.tcal
        ds.attrs['ical'] = self.ical
        ds.attrs['baudrate'] = self.baudrate
        ds.attrs['dev_structure_version'] = self.structure_version
        return ds


    def to_nc(self, out_filepath: os.path.abspath) -> None:
        """
        Export .dev data as a netcdf.

        :param out_filepath:
        """

        split = os.path.splitext(out_filepath)
        if split[-1] != '.nc':
            out_filepath += '.nc'
        ds = self.to_ds()
        ds.to_netcdf(out_filepath, engine='netcdf4')
