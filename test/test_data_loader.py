import logging
from unittest import TestCase

from tensorflow import TensorShape
from src.utils.config import Config
from src.data_loader.data_loader import DataLoader
from src.data_loader.preprocessing import Preprocess
import tensorflow as tf


class TestDataLoader(tf.test.TestCase):
    def setUp(self):
        self.data_loader = DataLoader()
        self.images = None
        self.masks = None
        config = {
            "images_path": "test/dummy_data/IMG/*.png",
            "masks_path": "test/dummy_data/MASK/*.png",
            "image_size": 256,
            "load_with_info": True,
            "output": 2,
            "output_files": "FINAL_5FC4_3CLUSTER_NCURSS1.3.3-K_NCU2_",
            "channel": 3
        }

        self.data = type('config', (object,), config)

    def test_not_none(self):
        [self.images, self.masks] = self.data_loader.load_data(self.data.images_path, self.data.masks_path, 2)
        self.assertIsNotNone(self.images)

    def test_same_number_images_masks(self):
        [self.images, self.masks] = self.data_loader.load_data(self.data.images_path, self.data.masks_path, 4)
        self.assertEqual(len(self.images), len(self.masks))

    def test_create_training_set(self):
        [self.images, self.masks] = self.data_loader.load_data(self.data.images_path, self.data.masks_path, 4)
        preprocess = Preprocess(1, 2, [0, 1, 2, 3, 4], self.data.image_size, self.images,
                                self.masks, self.images, self.masks)
        tr, val = preprocess.create_trainset()
        shape = (None, self.data.image_size, self.data.image_size, self.data.channel)
        self.assertItemsEqual(tr.element_spec[0].shape, shape)
        self.assertItemsEqual(val.element_spec[0].shape, shape)
        self.assertEqual(tr._batch_size, int(2))
        pass

    def test_fixx(self):
        self.test_same_number_images_masks()
        preprocess = Preprocess(1, 2, [0, 1, 2, 3, 4], self.data.image_size, self.images,
                                self.masks, self.images, self.masks)
        a = tf.constant(3, shape=[256, 256, 3])
        b = tf.constant(4, shape=[256, 256, 1])
        img_shape, mask_shape = preprocess.fixx(a, b)
        self.assertEqual(img_shape._shape, tf.TensorShape([256, 256, 3]))
        self.assertEqual(mask_shape._shape, tf.TensorShape([256, 256, 1]))

    def test_create_valds(self):
        [self.images, self.masks] = self.data_loader.load_data(self.data.images_path, self.data.masks_path, 4)
        preprocess = Preprocess(1, 2, [0, 1, 2, 3, 4], self.data.image_size, self.images,
                                self.masks, self.images, self.masks)
        val = preprocess.create_valds()
        shape = (None, self.data.image_size, self.data.image_size, self.data.channel)

        self.assertItemsEqual(val.element_spec[0].shape, shape)

    def test_create_testds(self):
        [self.images, self.masks] = self.data_loader.load_data(self.data.images_path, self.data.masks_path, 4)
        test = [x for x in range(len(self.images))]
        preprocess = Preprocess(1, 2, [0, 1, 2, 3, 4], self.data.image_size, self.images,
                                self.masks, self.images, self.masks)
        testing = preprocess.create_testds(test, 0)
        self.assertIsNotNone(testing, "Not none")

    def test_img_gen(self):
        pass
