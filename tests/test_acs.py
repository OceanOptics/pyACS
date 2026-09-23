import os
import unittest
from struct import unpack
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import interpolate
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # If tqdm is not available, use a dummy function


TEST_FRAME = b'\x53\xd0\x03\x01\x53\x00\x00\x02\x4e\x1e\x01\xba\x21\x29\x35\xff\
\x00\xff\x00\x02\xd0\x05\x01\x53\x00\x00\x02\x4e\x1a\x01\xba\x02\
\xa1\x7a\xe4\xb9\xd7\x01\xd5\x02\xb0\x00\x07\x1b\x02\x01\x56\x04\
\x05\x03\x63\x04\xf4\x03\x10\x04\x98\x03\xda\x05\xab\x03\xac\x05\
\x3b\x04\x69\x06\x7c\x04\x77\x05\xf5\x04\xfe\x07\x67\x05\x52\x06\
\xc6\x05\xae\x08\x69\x06\x4a\x07\x9f\x06\x66\x09\x88\x07\x3a\x08\
\x96\x07\x33\x0a\xc9\x08\x47\x09\xab\x08\x1b\x0c\x32\x09\x6a\x0a\
\xdc\x09\x0e\x0d\xc7\x0a\xa8\x0c\x2f\x0a\x2f\x0f\x84\x0c\x0f\x0d\
\x9a\x0b\x5c\x11\x63\x0d\x9a\x0f\x23\x0c\x9d\x13\x69\x0f\x3a\x10\
\xc5\x0d\xf8\x15\x8d\x11\x01\x12\x70\x0f\x5b\x17\xc9\x12\xd8\x14\
\x32\x10\xce\x1a\x25\x14\xd2\x16\x17\x12\x61\x1c\xb5\x16\xe8\x18\
\x20\x14\x08\x1f\x85\x19\x2b\x1a\x5b\x15\xe7\x22\x8e\x1b\xad\x1c\
\xbe\x17\xd8\x25\xc7\x1e\x5d\x1f\x3d\x19\xef\x29\x33\x21\x3b\x21\
\xe4\x1c\x17\x2c\xd5\x24\x48\x24\xb4\x1e\x6f\x30\xad\x27\x7f\x27\
\x9f\x20\xde\x34\xb9\x2a\xf7\x2a\xb6\x23\x65\x38\xf3\x2e\x8e\x2d\
\xe7\x26\x0b\x3d\x4e\x32\x5d\x31\x29\x28\xc6\x41\xd1\x36\x49\x34\
\x84\x2b\x82\x46\x63\x3a\x4f\x37\xd9\x2e\x49\x4a\xf4\x3e\x6a\x3b\
\x3b\x31\x17\x4f\xb2\x42\x96\x3e\xbc\x34\x09\x54\xa0\x46\xf8\x42\
\x6a\x37\x22\x59\xd2\x4b\x90\x46\x4e\x3a\x6c\x5f\x5a\x50\x7d\x4a\
\x75\x3d\xea\x65\x44\x55\xba\x4e\xf0\x41\xb3\x6b\x9a\x5b\x67\x53\
\xa2\x45\xb5\x72\x35\x61\x6f\x58\x73\x49\xbe\x78\xdf\x67\x9b\x5d\
\x41\x4d\xcb\x7f\x79\x6d\xd1\x61\xe2\x51\xab\x85\xd7\x73\xed\x66\
\x64\x55\x77\x8c\x10\x79\xf4\x6a\xbe\x59\x19\x92\x12\x7f\xcd\x6e\
\xef\x5c\xa0\x97\xd4\x85\x7d\x72\xeb\x60\x01\x9d\x63\x8b\x00\x76\
\xc2\x63\x41\xa2\xbd\x90\x5e\x7a\x22\x67\xe0\xa7\x92\x98\x2f\x7d\
\xb7\x6a\xc6\xac\x8c\x9d\x1a\x81\x13\x6d\x75\xb1\x37\xa1\xcc\x84\
\x35\x70\x05\xb5\xaa\xa6\x4f\x87\x25\x72\x54\xb9\xbf\xaa\x85\x89\
\xcd\x74\x63\xbd\x75\xae\x5d\x8c\x20\x76\x2b\xc0\xb9\xb1\xd0\x8e\
\x1e\x77\x95\xc3\x6e\xb4\xce\x8f\xaa\x78\x9c\xc5\x93\xb7\x3c\x90\
\xc0\x79\x3e\xc7\x05\xb9\x11\x91\x5e\x79\x8e\xc7\xe6\xba\x63\x91\
\x9c\x79\x82\xc8\x2d\xbb\x23\x91\x5d\x79\x13\xc7\xd9\xbb\x56\x90\
\xb5\x78\x5b\xc7\x00\xbb\x07\x8f\xac\x77\x2f\xc5\x83\xba\x1e\x8e\
\x28\x75\xc6\xc3\x76\xb8\xab\x8c\x52\x74\x0d\xc0\xf8\xb6\xc8\x8a\
\x1e\x72\x00\xbd\xe6\xb4\x5c\x87\x8e\x6f\xad\xba\x60\xb1\x6e\x84\
\xa9\x6d\x0a\xb6\x5c\xad\xf9\x81\x58\x6a\x07\xb1\xb7\xa9\xee\x7d\
\x9c\x66\xa8\xac\x6e\xa5\x37\x79\x72\x62\xeb\xa6\x8b\x9f\xe9\x74\
\xe3\x5e\xec\xa0\x25\x9a\x16\x70\x00\x5a\xc2\x99\x56\x93\xdb\x6a\
\xef\x56\x69\x92\x46\x8d\x4f\x65\xba\x52\x0f\x8b\x15\x86\x9e\x60\
\x82\x4d\xb2\x83\xdb\x7f\xd9\x5b\x53\x49\x5e\x7c\xb2\x79\x16\x56\
\x25\x45\x14\x75\x86\x72\x54\x51\x06\x40\xe1\x6e\x75\x6b\xb1\x4c\
\x13\x3c\xd0\x67\x9d\x65\x3a\x47\x3a\x38\xd5\x60\xf1\x5e\xde\x42\
\x89\x35\x02\x5a\x82\x58\xb8\x3e\x05\x31\x52\x54\x47\x52\xbf\x39\
\x9c\x2d\xbe\x4e\x3c\x4c\xfc\x35\x79\x2a\x70\x48\x9a\x47\x85\x31\
\x8c\x27\x36\x43\x35\x42\x52\x2d\xc8\x24\x33\x3e\x15\x3d\x5c\x2a\
\x3b\x21\x5a\x39\x39\x38\xa5\x26\xda\x1e\x9d\x34\x98\x34\x30\x23\
\xb3\x1c\x1c\x30\x50\x30\x03\x20\xbb\x19\xbf\x2c\x49\x2c\x1c\x22\
\x44\x00\xff\x00\xff\x00\x02\xd0\x03\x01\x53\x00\x00\x02\x4e\x1e'

TEST_COMPASS_DATA = 'test_data/'


class TestACSFunctions(unittest.TestCase):
    def test_compute_external_temperature(self):
        from pyACS.acs import ACS
        acs = ACS()
        counts = unpack('!H', b'\x7a\xe4')[0]
        su = acs.compute_external_temperature(counts)
        self.assertAlmostEqual(su, 22.14, 2)

    def test_compute_internal_temperature(self):
        from pyACS.acs import ACS
        acs = ACS()
        # AC-S (from ACS User Manual)
        counts = unpack('!H', b'\xb9\xd8')[0] # 16-bit unsigned integer == H (2 bytes unsigned short)
        su = acs.compute_internal_temperature(counts)
        self.assertAlmostEqual(su, 17.91, 2)

        # AC-S (from WetView User Manual (2004) error page 23-24)
        counts = unpack('!H', b'\xb9\xd7')[0]  # 16-bit unsigned integer == H (2 bytes unsigned short)
        su = acs.compute_internal_temperature(counts)
        self.assertAlmostEqual(su, 17.91, 2)

        # AC-9 (from WetView User Manual)
        # counts = unpack('!H', b'\x0f\x01')[0]
        # su = ac9.compute_internal_temperature()
        # self.assertAlmostEqual(su, 7.69, 2)

    def test_ac9_calibrate_math(self):
        # a_raw
        E_sig = 0.5356
        E_ref = 0.7813
        Z = 0.25
        a_raw = - np.log(E_sig / E_ref) / Z
        self.assertAlmostEqual(a_raw, 1.5103, 4)
        # compensation temperature
        T = 7.69
        T0 = 5.5233
        T1 = 8.4553
        delta_tn = 0.1411
        delta_tn1 = 0.1028
        delta_t = delta_tn + (T - T0) / (T1 - T0) * (delta_tn1 - delta_tn)
        self.assertAlmostEqual(delta_t, 0.11279, 4)
        delta_t = np.interp(T, np.array([T0, T1]), np.array([delta_tn, delta_tn1]))
        self.assertAlmostEqual(delta_t, 0.11279, 4)
        # Get a temperature calibrated
        a_tc = a_raw - delta_t
        self.assertAlmostEqual(a_tc, 1.3976, 3)
        # Add calibration offset
        C = 7.6242
        a = a_tc + C
        self.assertAlmostEqual(a, 9.0218, 3)

    def test_acs_calibrate_math(self):
        x = 0.25
        counts = unpack('!HHHH', b'\x04\x05\x03\x63\x04\xf4\x03\x10')
        c_ref, a_ref, c_sig, a_sig = counts[0], counts[1], counts[2], counts[3]
        c_raw = - (1 / x) * np.log(c_sig / c_ref)
        self.assertAlmostEqual(c_raw, -0.835, 3)
        a_raw = - (1 / x) * np.log(a_sig / a_ref)
        self.assertAlmostEqual(a_raw, 0.402, 2)
        # Discrepency in t_int in documentation 27.93 used in cal vs 17.91 computed (issue in manuals)
        internal_temperature_su = 27.93
        delta_t_a = np.interp(internal_temperature_su, [27.75, 28.2625], [0.010124, 0.015096])
        self.assertAlmostEqual(delta_t_a, 0.012, 3)
        offset_a = -0.431
        a = offset_a + a_raw - delta_t_a
        # self.assertAlmostEqual(a, -0.392, 3) # Error in manual 0.402 - 0.012 = 0.39 != 0.039
        self.assertAlmostEqual(a, -0.0409, 2)

    def test_acs_calibrate_frame(self):
        # Same as method test_acs_calibrate_math but use acs.calibrate_frame with 3 wavelength
        from pyACS import acs as pa
        # Define Frame
        counts = unpack('!HHHH', b'\x04\x05\x03\x63\x04\xf4\x03\x10')
        frame = pa.RawFrameContainer(serial_number='0x5300012A',  # acs 298
                                     output_wavelength=3,
                                     # t_int=unpack('!H', b'\xb9\xd7')[0],  # 17.91 Issue with temperature in wetview doc
                                     t_int=42969, # 42969=27.931 or 42970=27.929
                                     c_ref=counts[0],
                                     a_ref=np.array([counts[1],counts[1],counts[1]], dtype=np.uint16),
                                     c_sig=counts[2],
                                     a_sig=np.array([counts[3],counts[3],counts[3]], dtype=np.uint16),
                                     frame_len=np.nan,  # packet length
                                     frame_type=np.nan,  # Packet type identifier
                                     a_ref_dark=np.nan,  # A reference dark counts (for diagnostic purpose)
                                     p=np.nan,  # A/D counts from the pressure sensor circuitry
                                     a_sig_dark=np.nan,  # A signal dark counts (for diagnostic purpose)
                                     t_ext=np.nan,  # External temperature voltage counts
                                     c_ref_dark=np.nan,  # C reference dark counts
                                     c_sig_dark=np.nan,  # C signal dark counts
                                     timestamp=np.nan,  # unsigned integer: Instrument Timestamp (ms)
                                     checksum=np.nan,  # unsigned integer: Checksum (2 bytes)
                                     external_timestamp=np.nan)  # double: Logger timestamp (seconds)
        # Define ACS
        acs = pa.ACS()
        acs.serial_number = '0x5300012A'
        acs.output_wavelength = 3
        acs.t = np.array([27.75, 28.2625])
        acs.delta_t_c = np.empty((3,2))  # No value for testing
        acs.delta_t_a = np.empty((3,2))
        acs.delta_t_a[0,:] = np.array([0.010124, 0.015096])
        acs.delta_t_a[1,:] = np.array([0.010124, 0.015096])
        acs.delta_t_a[2,:] = np.array([0.010124, 0.015096])
        acs.offset_c = np.empty(3)
        acs.offset_a = np.array([-0.431,-0.431,-0.431])
        acs.x = 0.25

        # Finish init for SciPy runs
        acs.f_delta_t_c = interpolate.interp1d(acs.t, acs.delta_t_c, axis=1)
        acs.f_delta_t_a = interpolate.interp1d(acs.t, acs.delta_t_a, axis=1)

        # Test with SciPy
        pa.__dict__['SCIPY_IMPORTED'] = True
        cal = acs.calibrate_frame(frame)
        self.assertEqual(cal.flag_outside_calibration_range, False)
        for i in range(3):
            self.assertEqual(cal.a[0], cal.a[i])
            self.assertAlmostEqual(float(cal.a[i]), -0.0409, 2)

        # Test with numPy
        pa.__dict__['SCIPY_IMPORTED'] = False
        cal = acs.calibrate_frame(frame)
        self.assertEqual(cal.flag_outside_calibration_range, False)
        for i in range(3):
            self.assertEqual(cal.a[0], cal.a[i])
            self.assertAlmostEqual(float(cal.a[i]), -0.0409, 2)

    def test_acs_find_frame(self):
        from pyACS.acs import ACS

        acs = ACS()
        acs.output_wavelength = 86
        acs.set_frame_descriptor()

        frame, checksum, buffer, skipped = acs.find_frame(TEST_FRAME)
        self.assertEqual(frame, TEST_FRAME[15:738])  # Updated definitin=
        self.assertEqual(checksum, True)
        self.assertEqual(buffer, TEST_FRAME[738:])
        self.assertEqual(skipped, TEST_FRAME[:15])

    def test_acs_valid_frame(self):
        from pyACS.acs import ACS

        acs = ACS()
        acs.output_wavelength = 86
        acs.set_frame_descriptor()

        frame, checksum, buffer, skipped = acs.find_frame(TEST_FRAME)
        passed = acs.valid_frame(frame)
        self.assertEqual(passed, True)

    def test_acs_unpack_frame(self):
        from pyACS.acs import ACS

        acs = ACS()
        acs.output_wavelength = 86
        acs.set_frame_descriptor()

        frame, checksum, buffer, skipped = acs.find_frame(TEST_FRAME)
        data = acs.unpack_frame(frame)
        # Check header (from ACS Protocol (2009))
        self.assertEqual(data.frame_len, 720)
        self.assertEqual(data.frame_type, 5)
        self.assertEqual(data.serial_number, '0x53000002')
        self.assertEqual(data.timestamp, 465666)
        self.assertEqual(data.output_wavelength, 86)
        self.assertEqual(data.a_ref_dark, unpack('!H', b'\x4e\x1a')[0])
        self.assertEqual(data.p, unpack('!H', b'\x01\xba')[0])
        self.assertEqual(data.a_sig_dark, unpack('!H', b'\x02\xa1')[0])
        self.assertEqual(data.t_ext, unpack('!H', b'\x7a\xe4')[0])
        self.assertEqual(data.t_int, unpack('!H', b'\xb9\xd7')[0])
        self.assertEqual(data.c_ref_dark, unpack('!H', b'\x01\xd5')[0])
        self.assertEqual(data.c_sig_dark, unpack('!H', b'\x02\xb0')[0])
        # Check first counts
        self.assertEqual(data.c_ref[0], unpack('!H', b'\x04\x05')[0])
        self.assertEqual(data.a_ref[0], unpack('!H', b'\x03\x63')[0])
        self.assertEqual(data.c_sig[0], unpack('!H', b'\x04\xf4')[0])
        self.assertEqual(data.a_sig[0], unpack('!H', b'\x03\x10')[0])
        # Check last counts
        self.assertEqual(data.c_ref[-1], unpack('!H', b'\x20\xbb')[0])
        self.assertEqual(data.a_ref[-1], unpack('!H', b'\x19\xbf')[0])
        self.assertEqual(data.c_sig[-1], unpack('!H', b'\x2c\x49')[0])
        self.assertEqual(data.a_sig[-1], unpack('!H', b'\x2c\x1c')[0])

    def test_acs_check_data(self):
        from pyACS.acs import ACS

        acs = ACS()
        acs.serial_number = '0x53000002'
        acs.output_wavelength = 86
        acs.set_frame_descriptor()

        frame, checksum, buffer, skipped = acs.find_frame(TEST_FRAME)
        data = acs.unpack_frame(frame)
        passed = acs.check_data(data)
        self.assertEqual(passed, True)

    def _iter_dataset_parsers(self):
        """Yield (dataset_name, path_to_dataset, logging_software, parser, bin_files) for each valid dataset dir."""
        from pyACS.acs import ACSCompass, ACSInlinino

        datasets = [x for x in os.listdir(TEST_COMPASS_DATA) if os.path.isdir(os.path.join(TEST_COMPASS_DATA, x)) and '_ACS' in x]
        for d in tqdm(datasets):
            path_to_dataset = os.path.join(TEST_COMPASS_DATA, d)
            device_files = [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset) if f.endswith('.dev')]
            if not device_files:
                print('Skipping dataset %s: no .dev file found.' % d)
                continue
            if len(device_files) > 1:
                print('Skipping dataset %s: more than one .dev file found.' % d)
                continue
            device_file = device_files[0]
            logging_software = 'Inlinino' if 'Inlinino' in d else 'Compass'
            if logging_software == 'Inlinino':
                parser = ACSInlinino(device_file)
            elif logging_software == 'Compass':
                parser = ACSCompass(device_file)
            else:
                print('Skipping dataset %s: unknown logging software.' % d)
                continue
            bin_files = [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset) if f.endswith('.bin')]
            if not bin_files:
                print('Skipping dataset %s: no .bin file(s) found.' % d)
                continue
            yield d, path_to_dataset, logging_software, parser, bin_files

    def _load_truth_dataframe(self, logging_software, bin_file):
        """Locate and load the truth dataframe matching bin_file. Returns (None, None) if not found/supported."""
        if logging_software == 'Inlinino':
            csv_file = bin_file[:-4] + '.csv'
            if not os.path.isfile(csv_file):
                print('Skipping %s: no matching .csv file.' % bin_file)
                return None, None
            return read_inlinino_csv_output(csv_file), csv_file
        elif logging_software == 'Compass':
            dat_file = bin_file[:-4] + '.dat'
            if not os.path.isfile(dat_file):
                print('Skipping %s: no matching .dat file.' % bin_file)
                return None, None
            return read_prep_acs_output(dat_file), dat_file
        else:
            print('Unable to check, unknown logging software. %s' % bin_file)
            return None, None

    def _check_datasets(self, produce_actual_df):
        tested_files = 0
        for d, path_to_dataset, logging_software, parser, bin_files in self._iter_dataset_parsers():
            for bin_file in bin_files:
                actual_df, with_aux = produce_actual_df(logging_software, parser, bin_file)
                if len(actual_df) == 0:
                    print('No valid frames found for %s' % bin_file)
                    continue
                truth_df, truth_file = self._load_truth_dataframe(logging_software, bin_file)
                if truth_df is None:
                    continue
                # Assert loaded dataframe
                if logging_software == 'Inlinino':
                    self.assert_valid_inlinino_dataframe(actual_df, truth_df, with_aux=with_aux)
                elif logging_software == 'Compass':
                    self.assert_valid_compass_dataframe(actual_df, truth_df, with_aux=with_aux)
                else:
                    self.fail('Unknown logging software: %s' % logging_software)
                tested_files += 1

        self.assertGreater(tested_files, 0,
                           'No dataset had all .dev/.bin/.dat files present; test did not validate anything.')
        return tested_files

    # @unittest.skip("skipping Compass Dataset test")
    def test_reading_datasets(self):
        from pyACS.acs import BinToDataFrame

        def produce_actual_df(logging_software, parser, bin_file):
            actual_df = BinToDataFrame(parser).run(bin_file)
            with_aux = True  # aux is present in actual_df
            return actual_df, with_aux

        self._check_datasets(produce_actual_df)

    # @unittest.skip("skipping convert bin to csv")
    def test_convert_bin_to_csv(self):
        from pyACS.acs import ConvertBinToCSV

        def produce_actual_df(logging_software, parser, bin_file, write_aux=True):
            output_file = os.path.join(TEST_COMPASS_DATA, 'out.csv')  # os.path.basename(bin_file) + '_out.csv'
            if os.path.isfile(output_file):
                os.remove(output_file)
            ConvertBinToCSV(parser, bin_file, output_file, write_auxiliaries=write_aux)
            actual_df = pd.read_csv(output_file, delimiter=',') if os.path.isfile(output_file) else pd.DataFrame()
            return actual_df, write_aux

        self._check_datasets(produce_actual_df)

    def assert_valid_compass_dataframe(self, actual, truth, with_aux=True):
        # Check Time
        actual_time = actual['timestamp'] - actual['timestamp'][0]
        np.testing.assert_equal(actual_time.to_numpy(), truth['Time'].to_numpy())
        # Check Auxiliaries
        if with_aux:
            np.testing.assert_almost_equal(actual[['internal_temperature', 'external_temperature']].to_numpy(),
                                           truth[['int_temp', 'ext_temp']].to_numpy(), decimal=2)
        # Check c and a values
        actual_ca_idx = actual.columns.str.match(r'^[ac]\d')
        sel = actual.loc[:, actual_ca_idx] < 30
        np.testing.assert_almost_equal(actual.loc[:, actual_ca_idx].to_numpy()[sel],
                                       truth.iloc[:, 1:-6].to_numpy()[sel], decimal=4)

    def assert_valid_inlinino_dataframe(self, actual, truth, with_aux=True):
        # Check Timestamps
        np.testing.assert_equal(actual['timestamp'].to_numpy(), truth['acs_timestamp'].to_numpy())
        # Inlinino timestamps bug can create time offsets: 2026-04-01T15:13:18.999803 rounds to 15:13:18.000 instead of 15:13:19.000
        #   it could propagate to a minute, an hour, a day...
        delta = np.abs(actual['external_timestamp'].to_numpy().astype("datetime64[ms]")
                       - truth['time'].to_numpy().astype("datetime64[ms]"))
        # np.testing.assert_array_less(delta, np.timedelta64(1, 'ms'))  # Ideal testing
        np.testing.assert_array_less(delta, np.timedelta64(1, 'm'))  # due to bug in inlinino
        self.assertLessEqual(np.count_nonzero(delta > np.timedelta64(1, 'ms')),
                             0.05 * len(delta))  # due to bug in inlinino, need to tolerate some offsets.
        # Check Auxiliaries
        if with_aux:
            np.testing.assert_almost_equal(actual['internal_temperature'].to_numpy(), truth['T_int'].to_numpy(), decimal=2)
            np.testing.assert_almost_equal(actual['external_temperature'].to_numpy(), truth['T_ext'].to_numpy(), decimal=2)
        np.testing.assert_equal(actual['flag_outside_calibration_range'].to_numpy(),
                                truth['flag_outside_calibration_range'].to_numpy())
        # Check c and a values
        actual_ca_idx = actual.columns.str.match(r'^[ac]\d')
        np.testing.assert_almost_equal(actual.loc[:, actual_ca_idx].to_numpy(),
                                       truth.iloc[:, 5:].to_numpy(), decimal=4)

    def test_acs_repr(self):
        from pyACS.acs import ACS

        # Find data for test
        test_data_set = [x for x in os.listdir(TEST_COMPASS_DATA) if '_ACS' in x][0]
        path_to_dataset = os.path.join(TEST_COMPASS_DATA, test_data_set)
        device_file = [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset) if f.endswith('.dev')][0]

        # Test function works
        foo = repr(ACS(device_file))


def read_prep_acs_output(filename):
    df = pd.read_csv(filename, delimiter='\t')
    df.drop(['diagnostic', 'pr_acs'], axis=1, inplace=True)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    return df


def read_inlinino_csv_output(filename):
    # Read the first 3 rows to get metadata
    meta = pd.read_csv(filename, nrows=2, header=None)
    c_wavelengths = np.fromstring(meta.iloc[1, 2].split("lambda=")[1], sep=" ")
    a_wavelengths = np.fromstring(meta.iloc[1, 3].split("lambda=")[1], sep=" ")
    # Read data, skipping units and wavelength rows
    df = pd.read_csv(filename, skiprows=[1])
    # Parse time
    df["time"] = pd.to_datetime(df["time"])
    # Expand c and a arrays into individual columns
    c = pd.DataFrame(
        df.pop("c").apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).tolist(),
        columns=[f"c{w:.1f}" for w in c_wavelengths],
    )
    a = pd.DataFrame(
        df.pop("a").apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).tolist(),
        columns=[f"a{w:.1f}" for w in a_wavelengths],
    )
    return pd.concat([df, c, a], axis=1)


if __name__ == '__main__':
    unittest.main()
