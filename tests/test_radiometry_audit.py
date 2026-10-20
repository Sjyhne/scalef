import unittest
import numpy as np
from eval.radiometry_audit import audit_array, export_to_surface


class RadiometryAuditTest(unittest.TestCase):
    def test_offset_removed_once_and_dark_values_preserved(self):
        x = np.array([[[0, 0.05, 0.3]]], dtype=np.float32)
        np.testing.assert_allclose(export_to_surface(x, offset_mode="keep", offset=0.1),
                                   [[[0, -0.05, 0.2]]], atol=1e-7)
        np.testing.assert_array_equal(export_to_surface(x, offset_mode="remove", offset=0.1), x)
        with self.assertRaises(ValueError):
            export_to_surface(x, offset_mode="unknown", offset=0.1)

    def test_units_masks_and_strict_thresholds(self):
        x = np.array([[[10000., 15000., 16000.], [500., 0., 11000.], [90000., 90000., 90000.]]])
        mask = np.array([[True, True, False]])
        result = audit_array(x, mask, domain="dn", offset=0.1)
        self.assertEqual(result["finite_selected_pixels"], 2)
        self.assertEqual(result["above_dn_over_10000_1"]["count_per_band"], [0, 1, 2])
        self.assertEqual(result["above_dn_over_10000_1_5"]["count_per_band"], [0, 0, 1])
        self.assertEqual(result["above_surface_reflectance_1"]["count_per_band"], [0, 1, 1])
        equivalent = audit_array(x / 10000, mask, domain="dn_over_10000", offset=0.1)
        self.assertEqual(result, equivalent)

    def test_nonfinite_and_empty_support(self):
        x = np.array([[[np.nan, 0, 0], [0.1, 0.1, 0.1]]])
        r = audit_array(x, np.ones((1, 2)), domain="surface_reflectance", offset=0.1)
        self.assertEqual(r["nonfinite_selected_pixels"], 1)
        self.assertEqual(r["finite_selected_pixels"], 1)
        r = audit_array(x, np.zeros((1, 2)), domain="surface_reflectance", offset=0.1)
        self.assertEqual(r["status"], "empty_support")


if __name__ == "__main__":
    unittest.main()
