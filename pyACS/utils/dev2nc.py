from datetime import datetime, timezone
import netCDF4
import numpy as np
import os
import re

from pyACS.acs import ACS


class DEV2NC(ACS):

    """A basic utility class for converting an ACS dev file to a netCDF group."""
    def __init__(self, input_device_filepath, output_nc_filepath):
        """
        @param input_device_filepath: An absolute filepath of the ACS dev file desired to be converted.
        @param output_nc_filepath: An absolute filepath where you want to save the converted dev file to a netCDF.
        """

        super().__init__(os.path.abspath(input_device_filepath))
        self._savefp = os.path.abspath(output_nc_filepath)

        with open(input_device_filepath,'r') as dev_file:
            self._dev_lines = dev_file.readlines()

        self._get_offset_creation_date() # Get info not parsed in the original ACS class.
        self._get_tcal_ical()  # Get info not parsed in the original ACS class.
        self._get_noise_conform() # Get info not parsed in the original ACS class.
        self._to_nc()

    def _get_offset_creation_date(self):
        """
        Find the date the dev file was created so it can be added as a .nc attribute.
        """
        [loi] = [_line for _line in self._dev_lines if 'The offsets were saved' in _line] # Get line of interest.
        loi = loi.replace(' ','')
        self._dev_created_date = datetime.strptime(re.findall('on(.*?)\.',loi)[0], '%m/%d/%Y')


    def _get_tcal_ical(self):
        [loi] = [_line for _line in self._dev_lines if 'tcal:' in _line and 'ical:' in _line] # Get line of interest.
        self.tcal, self.ical = [float(v) for v in re.findall('tcal:(.*?)C, ical:(.*?) C',loi)[0]]


    def _get_noise_conform(self):
        [loi] = [_line for _line in self._dev_lines if 'maxANoise' in _line] # Get line of interest.
        values, params = loi.split(';')
        values = [float(v) for v in values.split('\t') if v != '']
        params = [str(v) for v in params.split('\t') if v != '']
        params = [p.replace(' ','') for p in params]
        params = [p.replace('\n','') for p in params]
        _dict =  dict(zip(params, values))
        for k, v in _dict.items():
            setattr(self,k,v)


    def _to_nc(self):
        """
        Take relevant information created by the ACS().read_device_file() function and create a netCDF group.
        """
        root = netCDF4.Dataset(self._savefp,'w', format = 'NETCDF4')
        calgroup = root.createGroup("factory_calibration")
        calgroup.title = "ACS Factory Calibration"
        calgroup.history = f"Factory Cal Date: {self._dev_created_date.strftime('%Y-%m-%dT%H:%M:%SZ')}\nConverted: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        calgroup.setncattr('factory_cal_date', self._dev_created_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        calgroup.version = '0.0.1'
        calgroup.setncattr('dev_structure_version', self.structure_version_number)
        calgroup.setncattr('serial_number', self.serial_number)
        calgroup.setncattr('baud_rate', self.baudrate)
        calgroup.setncattr('output_wavelengths', self.output_wavelength)
        calgroup.setncattr('output_temperatures', len(np.array(self.t)))
        calgroup.setncattr('path_length', self.x)
        calgroup.setncattr('depth_cal_offset', self.depth_cal_offset)
        calgroup.setncattr('depth_cal_scale_factor', self.depth_cal_scale_factor)
        calgroup.setncattr('tcal', self.tcal)
        calgroup.setncattr('ical', self.ical)
        calgroup.setncattr('max_a_noise', self.maxANoise)
        calgroup.setncattr('max_c_noise', self.maxCNoise)
        calgroup.setncattr('max_a_nonconform', self.maxANonConform)
        calgroup.setncattr('max_c_nonconform', self.maxCNonConform)
        calgroup.setncattr('max_a_difference', self.maxADifference)
        calgroup.setncattr('max_c_difference', self.maxCDifference)
        calgroup.setncattr('min_a_counts', self.minACounts)
        calgroup.setncattr('min_c_counts', self.minCCounts)
        calgroup.setncattr('min_r_counts', self.minRCounts)
        calgroup.setncattr('max_temp_sdev', self.maxTempSdev)
        calgroup.setncattr('max_depth_sdev', self.maxDepthSdev)

        temps = calgroup.createDimension('temperatures', None)
        _temps = calgroup.createVariable('temperatures','f8',("temperatures",))
        _temps[:] = np.array(self.t)

        awvls = calgroup.createDimension('wavelengths_a', None)
        _awvls = calgroup.createVariable('wavelengths_a','f8',("wavelengths_a",))
        _aoffs = calgroup.createVariable('offsets_a','f8',("wavelengths_a",))
        _adeltemps = calgroup.createVariable('delta_t_a','f8',("wavelengths_a","temperatures"))

        _awvls[:] = np.array(self.lambda_a)
        _aoffs[:] = np.array(self.offset_a)
        _adeltemps[:] = np.array(self.delta_t_a)

        cwvls = calgroup.createDimension('wavelengths_c', None)
        _cwvls = calgroup.createVariable('wavelengths_c','f8',("wavelengths_c",))
        _coffs = calgroup.createVariable('offsets_c','f8',("wavelengths_c",))
        _cdeltemps = calgroup.createVariable('delta_t_c','f8',("wavelengths_c","temperatures"))

        _cwvls[:] = np.array(self.lambda_c)
        _coffs[:] = np.array(self.offset_c)
        _cdeltemps[:] = np.array(self.delta_t_c)

        root.close()

        if os.path.exists(self._savefp):
            self.nc_path = os.path.abspath(self._savefp)
        else:
            raise FileNotFoundError(f"Failure to create {self._savefp}")