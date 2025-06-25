import unittest
import numpy as np
import pandas as pd

from iopmodel.IOP_models_func import (
    IOP_d_B22_C2,
    IOP_ph_B22_C2,
    calc_aph676,
    generate_aphs,
    rand_frac,
)

class TestIOPModelsFunc(unittest.TestCase):

    def test_IOP_d_B22_C2_basic(self):
        # Test with minimal required arguments
        result = IOP_d_B22_C2(ISM=1.0, Chl=2.0)
        self.assertIn("ad", result)
        self.assertIn("bd", result)
        self.assertIn("cd", result)
        self.assertIn("wavelen", result)
        self.assertTrue(np.all(result["bd"] >= 0))
        self.assertIsInstance(result["_parm"], dict)

    def test_IOP_d_B22_C2_custom_params(self):
        # Test with all parameters provided
        result = IOP_d_B22_C2(
            ISM=0.5, Chl=1.5,
            A=0.8, G=0.2,
            A_bd=0.01, S_bd=0.01, C_bd=0.01,
            A_md=0.01, S_md=0.01, C_md=0.01,
            qt_bd=0.5, qt_md=0.5,
            wavelen=np.array([400, 500, 550, 600, 700])
        )
        self.assertEqual(result["ad"].shape, (5,))
        self.assertEqual(result["bd"].shape, (5,))
        self.assertEqual(result["cd"].shape, (5,))

    def test_IOP_ph_B22_C2_basic(self):
        # Test with default arguments
        result = IOP_ph_B22_C2(Chl=1.0)
        self.assertIn("aph", result)
        self.assertIn("bph", result)
        self.assertIn("cph", result)
        self.assertIn("wavelen", result)
        self.assertEqual(result["aph"].shape, result["wavelen"].shape)
        self.assertEqual(result["bph"].shape, result["wavelen"].shape)
        self.assertEqual(result["cph"].shape, result["wavelen"].shape)

    def test_calc_aph676(self):
        # Test with scalar and array input
        out1 = calc_aph676(Chl=2.0)
        self.assertIn("values", out1)
        self.assertEqual(out1["values"].shape, ())
        out2 = calc_aph676(Chl=np.array([0.5, 2.0]))
        self.assertEqual(out2["values"].shape, (2,))

    def test_generate_aphs(self):
        # Test output structure
        out = generate_aphs(Chl=1.0)
        self.assertIn("aphs", out)
        self.assertIn("bphs", out)
        self.assertIn("cphs", out)
        self.assertIsInstance(out["aphs"], pd.DataFrame)
        self.assertIsInstance(out["bphs"], pd.DataFrame)
        self.assertIsInstance(out["cphs"], pd.DataFrame)

    def test_rand_frac(self):
        # Test output is a dict and sums to 1
        out = rand_frac()
        self.assertIsInstance(out, dict)
        total = sum(out.values())
        self.assertAlmostEqual(total, 1.0, places=5)

if __name__ == "__main__":
    unittest.main()