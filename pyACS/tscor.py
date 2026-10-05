import os
import xarray as xr


class TSCor():
    """
    A class for parsing ACS ts4.cor files.
    These files are often used to perform additional temperature and salinity correction.

    This class can be instantiated and data can be accessed as class attributes.

    Example Usage:
        tscor = TSCor('/home/jovyan/cals/ts4.cor')
        wavelengths = tscor.wavelengths

    Alternatively, the to_ds() function will format calibration data so that it can be used with
    operations based in xarray. The to_nc() function will export calibration data as a netcdf.
    """


    def __init__(self, filepath: os.path.abspath) -> None:
        """
        Parse the .cor file and assign data as attributes.

        :param filepath: The filepath of the TS4.cor file.
        """
        self.filepath = os.path.normpath(filepath)
        self.__read_cor()
        self.__parse_lines()

    def __read_cor(self):
        """Import the .cor file as a text file."""
        with open(self.filepath, 'r') as _file:
            self._lines = _file.readlines()

    def __parse_lines(self):
        """Parse the lines of the .cor file to get correction information."""

        wavelengths = []
        psi_t = []
        psi_s_c = []
        psi_s_a = []
        for line in self._lines:
            line_data = line.split('\t')
            line_data = [v.replace('\n', '') for v in line_data]
            line_data = [v.replace(' ', '') for v in line_data]
            if line_data == ['']:
                break
            line_data = [float(v) for v in line_data]
            wavelengths.append(line_data[0])
            psi_t.append(line_data[1])
            psi_s_c.append(line_data[2])
            psi_s_a.append(line_data[3])
        if len(wavelengths) != len(psi_t) != len(psi_s_c) != len(psi_s_a):
            raise ValueError('Mismatch in length of TS4cor file.')
        else:
            self.wavelengths = wavelengths
            self.psi_t = psi_t
            self.psi_s_c = psi_s_c
            self.psi_s_a = psi_s_a

    def to_ds(self) -> xr.Dataset:
        """
        Export class attributes to an xarray dataset.

        :return: An xarray dataset containing correction data.
        """
        ds = xr.Dataset()
        ds = ds.assign_coords({'wavelength': self.wavelengths})
        ds['psi_t'] = (['wavelength'], self.psi_t)
        ds['psi_s_c'] = (['wavelength'], self.psi_s_c)
        ds['psi_s_a'] = (['wavelength'], self.psi_s_a)
        return ds

    def to_nc(self, out_filepath: os.path.abspath) -> None:
        """
        Export .cor data as a netcdf.

        :param out_filepath: The save location of netcdf containing .cor information.
        """

        split = os.path.splitext(out_filepath)
        if split[-1] != '.nc':
            out_filepath += '.nc'
        ds = self.to_ds()
        ds.to_netcdf(out_filepath, engine='netcdf4')
