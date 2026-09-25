import unittest

import numpy as np

from image_metrics import compare_images, mse, normalized_rmse


class ImageMetricsTest(unittest.TestCase):
    def test_known_error(self):
        reference = np.full((2, 3, 3), 2.0, dtype=np.float32)
        image = np.full_like(reference, 3.0)
        self.assertEqual(mse(image, reference), 1.0)
        self.assertEqual(normalized_rmse(image, reference), 0.5)

    def test_identical_and_black_images(self):
        image = np.zeros((1, 2, 3), dtype=np.float32)
        self.assertEqual(compare_images(image, image), {"mse": 0.0, "normalized_rmse": 0.0})
        self.assertTrue(np.isinf(normalized_rmse(np.ones_like(image), image)))

    def test_no_clipping_or_integer_underflow(self):
        image = np.array([[[-2.0, 0.0, 4.0]]], dtype=np.float32)
        self.assertEqual(mse(image, np.zeros_like(image)), 20.0 / 3.0)
        low = np.zeros((1, 1, 3), dtype=np.uint8)
        high = np.full_like(low, 255)
        self.assertEqual(mse(low, high), 255.0 ** 2)

    def test_float64_reduction(self):
        image = np.full((2, 2, 3), 1e20, dtype=np.float32)
        result = mse(image, np.zeros_like(image))
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result / float(image[0, 0, 0]) ** 2, 1.0)

    def test_invalid_dimensions(self):
        for shape in ((1, 3), (2, 2, 4), (0, 2, 3), (2, 0, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                mse(np.zeros(shape), np.zeros(shape))
        with self.assertRaises(ValueError):
            mse(np.zeros((1, 2, 3)), np.zeros((2, 1, 3)))

    def test_nonfinite_values(self):
        valid = np.ones((2, 2, 3))
        for value in (np.nan, np.inf, -np.inf):
            invalid = valid.copy()
            invalid[0, 0, 0] = value
            for image, reference in ((invalid, valid), (valid, invalid)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    mse(image, reference)

    def test_invalid_types(self):
        for dtype in (object, complex, str, bool):
            image = np.ones((1, 1, 3), dtype=dtype)
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                mse(image, image)


if __name__ == "__main__":
    unittest.main()
