import unittest
import tensorflow as tf
import numpy as np
from src.model.parcel_based_kappa_with_area_weights import change_pixel_value_of_region, \
    get_pixel_counts_of_target_region_in_model_result, \
    calculate_parcel_based_kappa_with_area_weights


class MyTestCase(tf.test.TestCase):
    def setUp(self):
        self.pixel_based_model_result_path = r"test/dummy_data/pixeled/cropped_pixel.png"
        self.parcel_mask_path = r"test/dummy_data/parceled/cropped_parcel.png"

    def test_1(self):
        image_array = np.load('test/dummy_data/parcel_based_model_result.npy')
        coordinates = np.load('test/dummy_data/coords.npy')
        target_value = 1
        image_array = change_pixel_value_of_region(image_array, coordinates, target_value)
        self.assertEqual(len(np.unique(image_array)), 2)

    def test_2(self):
        model_result_image_array = np.load('test/dummy_data/pixel_based_model_result.npy')
        coordinates = np.load('test/dummy_data/coords.npy')
        target_value = 255
        count = get_pixel_counts_of_target_region_in_model_result(model_result_image_array, coordinates, target_value)
        self.assertEqual(count, 9737)

    def test_4(self):
        parcel_based_kappa_with_area_weight = calculate_parcel_based_kappa_with_area_weights(self.parcel_mask_path,
                                                                                             self.pixel_based_model_result_path,
                                                                                             512, 512,
                                                                                             ratio_threshold=0.5)
        self.assertEqual(parcel_based_kappa_with_area_weight, 1)
