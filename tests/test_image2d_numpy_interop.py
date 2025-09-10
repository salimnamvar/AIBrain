"""Comprehensive tests for Image2D numpy interoperability.

This module tests the Image2D class's integr        # Test shape manipulation
        flattened = self.gray_img.flatten()
        self.assertIsInstance(flattened, Image2D)
        np.testing.assert_array_equal(flattened.data, self.gray_image_data.flatten())n with numpy, OpenCV, scipy,
and other numerical libraries to ensure seamless interoperability.
"""

import unittest
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
import torch
from scipy import ndimage
from scipy.signal import convolve2d

from aib.cv.geom.size import Size
from aib.cv.img.image import Image2D


class TestImage2DNumpyInteroperability(unittest.TestCase):
    """Test suite for Image2D numpy interoperability."""

    def setUp(self):
        """Set up test fixtures."""
        # Create various test images
        self.gray_image_data = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        self.rgb_image_data = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        self.float_image_data = np.random.rand(100, 150).astype(np.float32)
        self.rgba_image_data = np.random.randint(0, 256, (100, 150, 4), dtype=np.uint8)

        # Create Image2D instances
        self.gray_img = Image2D(data=self.gray_image_data, filename="test_gray.jpg")
        self.rgb_img = Image2D(data=self.rgb_image_data, filename="test_rgb.jpg")
        self.float_img = Image2D(data=self.float_image_data, filename="test_float.tif")
        self.rgba_img = Image2D(data=self.rgba_image_data, filename="test_rgba.png")

    def test_basic_numpy_array_conversion(self):
        """Test basic numpy array conversion using np.asarray and np.array."""
        # Test np.asarray
        gray_array = np.asarray(self.gray_img)
        np.testing.assert_array_equal(gray_array, self.gray_image_data)
        self.assertEqual(gray_array.dtype, self.gray_image_data.dtype)

        rgb_array = np.asarray(self.rgb_img)
        np.testing.assert_array_equal(rgb_array, self.rgb_image_data)

        # Test np.array
        gray_copy = np.array(self.gray_img)
        np.testing.assert_array_equal(gray_copy, self.gray_image_data)

        # Test dtype conversion
        float_gray = np.array(self.gray_img, dtype=np.float32)
        expected_float = self.gray_image_data.astype(np.float32)
        np.testing.assert_array_equal(float_gray, expected_float)

    def test_numpy_ufunc_operations(self):
        """Test numpy universal functions work correctly with Image2D."""
        # Arithmetic operations
        result_add = np.add(self.gray_img, 50)
        self.assertIsInstance(result_add, Image2D)
        np.testing.assert_array_equal(result_add.data, self.gray_image_data + 50)
        self.assertEqual(result_add.filename, "test_gray.jpg")

        result_multiply = np.multiply(self.float_img, 2.0)
        self.assertIsInstance(result_multiply, Image2D)
        np.testing.assert_array_equal(result_multiply.data, self.float_image_data * 2.0)

        # Mathematical functions
        result_sqrt = np.sqrt(self.float_img)
        self.assertIsInstance(result_sqrt, Image2D)
        np.testing.assert_array_equal(result_sqrt.data, np.sqrt(self.float_image_data))

        # Trigonometric functions
        result_sin = np.sin(self.float_img)
        self.assertIsInstance(result_sin, Image2D)
        np.testing.assert_array_equal(result_sin.data, np.sin(self.float_image_data))

        # Comparison operations
        result_greater = np.greater(self.gray_img, 128)
        self.assertIsInstance(result_greater, Image2D)
        np.testing.assert_array_equal(result_greater.data, self.gray_image_data > 128)

    def test_numpy_array_functions(self):
        """Test numpy array functions work with Image2D."""
        # Statistical functions
        mean_val = np.mean(self.gray_img)
        expected_mean = np.mean(self.gray_image_data)
        self.assertAlmostEqual(mean_val, expected_mean)

        std_val = np.std(self.rgb_img)
        expected_std = np.std(self.rgb_image_data)
        self.assertAlmostEqual(std_val, expected_std)

        # Axis-specific operations
        mean_axis0 = np.mean(self.rgb_img, axis=0)
        self.assertIsInstance(mean_axis0, Image2D)
        np.testing.assert_array_equal(mean_axis0.data, np.mean(self.rgb_image_data, axis=0))

        # Reduction operations
        max_val = np.max(self.gray_img)
        expected_max = np.max(self.gray_image_data)
        self.assertEqual(max_val, expected_max)

        # Shape manipulation
        flattened = self.gray_img.flatten()
        self.assertIsInstance(flattened, Image2D)
        np.testing.assert_array_equal(flattened.data, self.gray_image_data.flatten())

    def test_opencv_interoperability(self):
        """Test OpenCV operations with Image2D."""
        # Basic OpenCV operations
        # Gaussian blur - convert to numpy array first
        blurred_cv = cv2.GaussianBlur(np.array(self.rgb_img), (5, 5), 1.0)
        blurred_img = Image2D(data=blurred_cv, filename=self.rgb_img.filename)
        self.assertIsInstance(blurred_img, Image2D)
        expected_blur = cv2.GaussianBlur(self.rgb_image_data, (5, 5), 1.0)
        np.testing.assert_array_equal(blurred_img.data, expected_blur)

        # Color space conversion
        if self.rgb_img.channels == 3:
            gray_cv = cv2.cvtColor(np.array(self.rgb_img), cv2.COLOR_RGB2GRAY)
            gray_img = Image2D(data=gray_cv, filename=self.rgb_img.filename)
            self.assertIsInstance(gray_img, Image2D)
            expected_gray = cv2.cvtColor(self.rgb_image_data, cv2.COLOR_RGB2GRAY)
            np.testing.assert_array_equal(gray_img.data, expected_gray)

        # Edge detection
        edges = cv2.Canny(np.array(self.gray_img), 50, 150)
        edges_img = Image2D(data=edges, filename=self.gray_img.filename)
        self.assertIsInstance(edges_img, Image2D)
        expected_edges = cv2.Canny(self.gray_image_data, 50, 150)
        np.testing.assert_array_equal(edges_img.data, expected_edges)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(np.array(self.gray_img), kernel, iterations=1)
        eroded_img = Image2D(data=eroded, filename=self.gray_img.filename)
        self.assertIsInstance(eroded_img, Image2D)
        expected_eroded = cv2.erode(self.gray_image_data, kernel, iterations=1)
        np.testing.assert_array_equal(eroded_img.data, expected_eroded)

        # Histogram calculation
        hist = cv2.calcHist([np.array(self.gray_img)], [0], None, [256], [0, 256])
        expected_hist = cv2.calcHist([self.gray_image_data], [0], None, [256], [0, 256])
        np.testing.assert_array_equal(hist, expected_hist)

    def test_scipy_interoperability(self):
        """Test SciPy operations with Image2D."""
        # ndimage operations - scipy doesn't recognize Image2D, so we convert and wrap
        rotated = ndimage.rotate(np.array(self.float_img), 45, reshape=False)
        rotated_img = Image2D(data=rotated, filename=self.float_img.filename)
        self.assertIsInstance(rotated_img, Image2D)
        expected_rotated = ndimage.rotate(self.float_image_data, 45, reshape=False)
        np.testing.assert_array_almost_equal(rotated_img.data, expected_rotated)

        # Gaussian filter
        filtered = ndimage.gaussian_filter(np.array(self.float_img), sigma=1.0)
        filtered_img = Image2D(data=filtered, filename=self.float_img.filename)
        self.assertIsInstance(filtered_img, Image2D)
        expected_filtered = ndimage.gaussian_filter(self.float_image_data, sigma=1.0)
        np.testing.assert_array_almost_equal(filtered_img.data, expected_filtered)

        # Zoom operation
        zoomed = ndimage.zoom(np.array(self.gray_img), 0.5)
        zoomed_img = Image2D(data=zoomed, filename=self.gray_img.filename)
        self.assertIsInstance(zoomed_img, Image2D)
        expected_zoomed = ndimage.zoom(self.gray_image_data, 0.5)
        np.testing.assert_array_equal(zoomed_img.data, expected_zoomed)

        # Convolution with scipy.signal
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        if self.gray_img.data.ndim == 2:
            convolved = convolve2d(np.array(self.gray_img), kernel, mode='same', boundary='symm')
            convolved_img = Image2D(data=convolved, filename=self.gray_img.filename)
            self.assertIsInstance(convolved_img, Image2D)
            expected_conv = convolve2d(self.gray_image_data, kernel, mode='same', boundary='symm')
            np.testing.assert_array_equal(convolved_img.data, expected_conv)

    def test_pytorch_interoperability(self):
        """Test PyTorch tensor conversion and operations."""
        # Convert to PyTorch tensor
        tensor_gray = torch.from_numpy(np.array(self.gray_img))
        self.assertIsInstance(tensor_gray, torch.Tensor)
        np.testing.assert_array_equal(tensor_gray.numpy(), self.gray_image_data)

        tensor_float = torch.from_numpy(np.array(self.float_img))
        self.assertIsInstance(tensor_float, torch.Tensor)
        np.testing.assert_array_almost_equal(tensor_float.numpy(), self.float_image_data)

        # PyTorch operations
        tensor_rgb = torch.from_numpy(np.array(self.rgb_img)).float()

        # Test tensor operations that return numpy arrays
        tensor_mean = torch.mean(tensor_rgb, dim=2)
        mean_back_to_image = Image2D(data=tensor_mean.numpy())
        self.assertIsInstance(mean_back_to_image, Image2D)

        # Test tensor normalization
        normalized_tensor = (tensor_rgb - tensor_rgb.min()) / (tensor_rgb.max() - tensor_rgb.min())
        normalized_image = Image2D(data=normalized_tensor.numpy())
        self.assertIsInstance(normalized_image, Image2D)

    def test_tensorflow_interoperability(self):
        """Test TensorFlow tensor conversion and operations."""
        # Convert to TensorFlow tensor
        tf_tensor_gray = tf.constant(np.array(self.gray_img))
        self.assertIsInstance(tf_tensor_gray, tf.Tensor)
        np.testing.assert_array_equal(tf_tensor_gray.numpy(), self.gray_image_data)

        tf_tensor_float = tf.constant(np.array(self.float_img))
        np.testing.assert_array_almost_equal(tf_tensor_float.numpy(), self.float_image_data)

        # TensorFlow operations
        tf_tensor_rgb = tf.constant(np.array(self.rgb_img), dtype=tf.float32)

        # Test tensor operations
        tf_mean = tf.reduce_mean(tf_tensor_rgb, axis=2)
        mean_back_to_image = Image2D(data=tf_mean.numpy())
        self.assertIsInstance(mean_back_to_image, Image2D)

        # Test image processing operations
        if len(tf_tensor_rgb.shape) == 3:
            # Add batch dimension for TF image ops
            batched_tensor = tf.expand_dims(tf_tensor_rgb, 0)
            resized = tf.image.resize(batched_tensor, [50, 75])
            resized_squeezed = tf.squeeze(resized, 0)
            resized_image = Image2D(data=resized_squeezed.numpy().astype(np.uint8))
            self.assertIsInstance(resized_image, Image2D)
            self.assertEqual(resized_image.height, 50)
            self.assertEqual(resized_image.width, 75)

    def test_array_interface_protocol(self):
        """Test the array interface protocol."""
        # Test __array_interface__
        interface = self.gray_img.__array_interface__
        self.assertIsInstance(interface, dict)
        self.assertIn('shape', interface)
        self.assertIn('typestr', interface)
        self.assertIn('data', interface)

        # Verify interface matches underlying data
        self.assertEqual(interface['shape'], self.gray_image_data.shape)
        self.assertEqual(interface['typestr'], self.gray_image_data.dtype.str)

    def test_indexing_and_slicing(self):
        """Test that indexing and slicing preserve Image2D type."""
        # Test slicing
        roi = self.rgb_img[10:50, 20:80]
        self.assertIsInstance(roi, Image2D)
        np.testing.assert_array_equal(roi.data, self.rgb_image_data[10:50, 20:80])
        self.assertEqual(roi.filename, "test_rgb.jpg")

        # Test advanced indexing
        mask = self.gray_img > 128
        masked = self.gray_img[mask.data]
        self.assertIsInstance(masked, Image2D)

        # Test single element access
        single_row = self.rgb_img[0]
        self.assertIsInstance(single_row, Image2D)
        np.testing.assert_array_equal(single_row.data, self.rgb_image_data[0])

    def test_concatenation_and_stacking(self):
        """Test numpy concatenation and stacking operations."""
        # Create another image for concatenation
        img2_data = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        img2 = Image2D(data=img2_data, filename="test2.jpg")

        # Test horizontal concatenation
        h_concat = np.concatenate([self.gray_img, img2], axis=1)
        self.assertIsInstance(h_concat, Image2D)
        expected_h_concat = np.concatenate([self.gray_image_data, img2_data], axis=1)
        np.testing.assert_array_equal(h_concat.data, expected_h_concat)

        # Test vertical concatenation
        v_concat = np.concatenate([self.gray_img, img2], axis=0)
        self.assertIsInstance(v_concat, Image2D)
        expected_v_concat = np.concatenate([self.gray_image_data, img2_data], axis=0)
        np.testing.assert_array_equal(v_concat.data, expected_v_concat)

        # Test stacking
        stacked = np.stack([self.gray_img, img2], axis=0)
        self.assertIsInstance(stacked, Image2D)
        expected_stacked = np.stack([self.gray_image_data, img2_data], axis=0)
        np.testing.assert_array_equal(stacked.data, expected_stacked)

    def test_linear_algebra_operations(self):
        """Test linear algebra operations with numpy.linalg."""
        # Create a square image for matrix operations
        square_data = np.random.rand(50, 50).astype(np.float32)
        square_img = Image2D(data=square_data)

        # Test matrix operations
        # Note: These operations might not preserve Image2D type if they change dimensionality
        det = np.linalg.det(square_img)
        expected_det = np.linalg.det(square_data)
        self.assertAlmostEqual(det, expected_det, places=5)

        # Test operations that preserve shape
        transposed = np.transpose(square_img)
        self.assertIsInstance(transposed, Image2D)
        np.testing.assert_array_equal(transposed.data, np.transpose(square_data))

    def test_broadcasting_operations(self):
        """Test numpy broadcasting with Image2D."""
        # Test broadcasting with scalars
        scaled = self.float_img * 2.0
        self.assertIsInstance(scaled, Image2D)
        np.testing.assert_array_equal(scaled.data, self.float_image_data * 2.0)

        # Test broadcasting with arrays
        if self.rgb_img.channels == 3:
            # Broadcast with color correction vector
            color_correction = np.array([1.1, 0.9, 1.0])
            corrected = self.rgb_img * color_correction
            self.assertIsInstance(corrected, Image2D)
            expected_corrected = self.rgb_image_data * color_correction
            np.testing.assert_array_equal(corrected.data, expected_corrected)

    def test_memory_views_and_copying(self):
        """Test memory views and copying behavior."""
        # Test that to_numpy returns a view by default
        numpy_view = self.gray_img.to_numpy()
        self.assertTrue(np.shares_memory(numpy_view, self.gray_img.data))

        # Test copying
        numpy_copy = np.copy(self.gray_img)
        self.assertIsInstance(numpy_copy, Image2D)
        self.assertFalse(np.shares_memory(numpy_copy.data, self.gray_img.data))
        np.testing.assert_array_equal(numpy_copy.data, self.gray_img.data)

    def test_dtype_conversions(self):
        """Test various dtype conversions."""
        # Convert uint8 to float32
        float_converted = self.gray_img.astype(np.float32)
        self.assertIsInstance(float_converted, Image2D)
        self.assertEqual(float_converted.data.dtype, np.float32)
        np.testing.assert_array_equal(float_converted.data, self.gray_image_data.astype(np.float32))

        # Convert float to uint8
        uint8_converted = self.float_img.astype(np.uint8)
        self.assertIsInstance(uint8_converted, Image2D)
        self.assertEqual(uint8_converted.data.dtype, np.uint8)

    def test_image_properties_preservation(self):
        """Test that Image2D properties are preserved through operations."""
        # Test that filename is preserved through numpy operations
        result = np.add(self.gray_img, 10)
        self.assertEqual(result.filename, "test_gray.jpg")

        # Test size properties
        self.assertEqual(self.rgb_img.width, 150)
        self.assertEqual(self.rgb_img.height, 100)
        self.assertEqual(self.rgb_img.channels, 3)

        # Test that properties work after operations
        scaled = self.rgb_img * 0.5
        self.assertEqual(scaled.width, 150)
        self.assertEqual(scaled.height, 100)
        self.assertEqual(scaled.channels, 3)

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test with incompatible types
        with self.assertRaises((TypeError, ValueError)):
            # This should fail as string cannot be broadcast with image
            _ = self.gray_img + "invalid"

        # Test invalid indexing
        with self.assertRaises(IndexError):
            _ = self.gray_img[1000, 1000]  # Out of bounds


if __name__ == '__main__':
    unittest.main()
