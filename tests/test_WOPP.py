import unittest
import numpy as np
from iopmodel.WOPP import WOPP

class TestWOPP(unittest.TestCase):
    def setUp(self):
        # Example wavelengths in nm
        self.wavelen = np.array([400, 500, 600, 700, 800, 900, 1000])
        self.default_temp = 25
        self.default_sal = 10

    def test_output_keys(self):
        result = WOPP(Temp=self.default_temp, Sal=self.default_sal, wavelen=self.wavelen)
        expected_keys = {'wavelen', 'a', 'b', 'Temp', 'Sal', 'nw', 'aw_version'}
        self.assertTrue(expected_keys.issubset(result.keys()))

    def test_output_shapes(self):
        result = WOPP(Temp=self.default_temp, Sal=self.default_sal, wavelen=self.wavelen)
        n = len(self.wavelen)
        self.assertEqual(result['a'].shape, (n,))
        self.assertEqual(result['b'].shape, (n,))
        self.assertEqual(result['nw'].shape, (n,))
        np.testing.assert_array_equal(result['wavelen'], self.wavelen)

    def test_aw_version(self):
        for version in [1, 2, 3]:
            result = WOPP(Temp=self.default_temp, Sal=self.default_sal, wavelen=self.wavelen, aw_version=version)
            self.assertEqual(result['aw_version'], version)
        with self.assertRaises(ValueError):
            WOPP(Temp=self.default_temp, Sal=self.default_sal, wavelen=self.wavelen, aw_version=4)

    def test_temp_and_salinity_effect(self):
        # Changing temperature or salinity should affect the output
        result1 = WOPP(Temp=0, Sal=0, wavelen=self.wavelen)
        result2 = WOPP(Temp=30, Sal=35, wavelen=self.wavelen)
        self.assertFalse(np.allclose(result1['a'], result2['a']))
        self.assertFalse(np.allclose(result1['b'], result2['b']))
        self.assertFalse(np.allclose(result1['nw'], result2['nw']))

    def test_missing_wavelen_raises(self):
        with self.assertRaises(ValueError):
            WOPP(Temp=self.default_temp, Sal=self.default_sal)

if __name__ == '__main__':
    unittest.main()