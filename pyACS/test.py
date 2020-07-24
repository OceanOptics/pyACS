import unittest
from struct import unpack
import numpy as np
from scipy import interpolate

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


class TestACSFunctions(unittest.TestCase):

    # Called before each individual test function is run
    # def setUp(self):
    #     pass

    # Called after the test
    # def tearDown(self):
    #     pass

    # Methods with test_ as prefix are run as test
    def test_compute_external_temperature(self):
        from pyACS import acs
        counts = unpack('!H', b'\x7a\xe4')[0]
        su = acs.compute_external_temperature(counts)
        self.assertAlmostEqual(su, 22.14, 2)

    def test_compute_internal_temperature(self):
        from pyACS import acs
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
        frame = pa.FrameContainer(serial_number='0x5300012A',  # acs 298
                                   output_wavelength=3,
                                   # t_int=unpack('!H', b'\xb9\xd7')[0],  # 17.91 Issue with temperature in wetview doc
                                   t_int=42969, # 42969=27.931 ou 42970=27.929
                                   c_ref=counts[0],
                                   a_ref=np.array([counts[1],counts[1],counts[1]], dtype=np.uint16),
                                   c_sig=counts[2],
                                   a_sig=np.array([counts[3],counts[3],counts[3]], dtype=np.uint16),
                                   frame_len=np.NaN,  # packet length
                                   frame_type=np.NaN,  # Packet type identifier
                                   a_ref_dark=np.NaN,  # A reference dark counts (for diagnostic purpose)
                                   p=np.NaN,  # A/D counts from the pressure sensor circuitry
                                   a_sig_dark=np.NaN,  # A signal dark counts (for diagnostic purpose)
                                   t_ext=np.NaN,  # External temperature voltage counts
                                   c_ref_dark=np.NaN,  # C reference dark counts
                                   c_sig_dark=np.NaN,  # C signal dark counts
                                   time_stamp=np.NaN  # unsigned integer: Time stamp (ms)
                                   )
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
        _, a, flag_outside_temperature_calibration_range = acs.calibrate_frame(frame)
        self.assertEqual(flag_outside_temperature_calibration_range, False)
        for i in range(3):
            self.assertEqual(a[0], a[i])
            self.assertAlmostEqual(float(a[i]), -0.0409, 2)

        # Test with numPy
        pa.__dict__['SCIPY_IMPORTED'] = False
        _, a, flag_outside_temperature_calibration_range = acs.calibrate_frame(frame)
        self.assertEqual(flag_outside_temperature_calibration_range, False)
        for i in range(3):
            self.assertEqual(a[0], a[i])
            self.assertAlmostEqual(float(a[i]), -0.0409, 2)

    def test_acs_unpack_frame(self):
        from pyACS.acs import BinReader, ACS, FrameIncompleteError

        acs = ACS()
        acs.serial_number = '0x53000002'
        acs.set_output_wavelength(86)
        acs.frame_type = 5

        class BinReaderTest(BinReader):
            def handle_frame(self2, frame):
                try:
                    data = acs.unpack_frame(frame)
                    # Check header (from ACS Protocol (2009))
                    self.assertEqual(data.frame_len, 720)
                    self.assertEqual(data.frame_type, 5)
                    self.assertEqual(data.time_stamp, 465666)
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
                except FrameIncompleteError:
                    pass

        b = BinReaderTest()
        b.data_read(TEST_FRAME)


if __name__ == '__main__':
    unittest.main()
