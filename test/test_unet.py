import unittest
from src.model.unet import Unet
import numpy as np
import tensorflow as tf
import os


# from test.test_data_loader import TestDataLoader
# from test.code_style_check import TestCodeFormat

class TestUnet(tf.test.TestCase):
    def setUp(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        CFG = {
            "data": {
                "images_path": "test/dummy_data/IMG/*.png",
                "masks_path": "test/dummy_data/MASK/*.png",
                "test_images_path": "test/dummy_data/testing_img/*.png",
                "test_masks_path": "test/dummy_data/testing_mask/*.png",
                "image_size": 256,
                "load_with_info": True
            },
            "train": {
                "batch_size": 2,
                "buffer_size": 1000,
                "epoches": 2,
                "val_subsplits": 5,
                "optimizer": {
                    "type": "adam"
                },
                "metrics": ["accuracy"]
            },
            "model": {
                "input": [256, 256, 3],  # modify this to our model
                "up_stack": {
                    "layer_1": 512,
                    "layer_2": 256,
                    "layer_3": 128,
                    "layer_4": 64,
                    "kernels": 3
                },
                "output": 3
            }
        }

        self.unet = Unet(CFG, 1)

    def tearDown(self):
        self.unet.f_width = 48
        self.unet.f_height = 45
        if os.path.exists('U_BS2_I256_note.h5'):
            os.remove('U_BS2_I256_note.h5')
        else:
            pass

    def test_load_data(self):
        self.unet.load_data(4)
        shape = (None, self.unet.image_size, self.unet.image_size, self.unet.output_channels)
        self.assertItemsEqual(self.unet.train_dataset.element_spec[0].shape, shape)
        self.assertItemsEqual(self.unet.val_ds.element_spec[0].shape, shape)

    def test_build(self):
        self.unet.build()
        output_layer = self.unet.model.get_layer('output').output_shape
        input_layer = self.unet.model.get_layer('input').output_shape[0]
        self.assertIsNotNone(self.unet.model)
        self.assertEqual(output_layer, input_layer)

    def test__upsample(self):
        self.assertEqual(len(self.unet._upsample(512, (3, 3))), 3)

    def test_train(self):
        """it will not be done in this github action require a lot of memory"""
        # run one step training and check whether the weight is updated or not
        # check that the loss is reduce after one batch
        # To do But HOW?
        self.unet.load_data(4)
        self.unet.build()
        history = self.unet.train()
        self.assertAllLess(history[0][1], history[0][0])

    def test_evaluate(self):
        """it will not be done in this github action require a lot of memory"""
        self.unet.load_data(4)
        self.unet.build()
        self.unet.train()
        self.unet.f_width = 2
        self.unet.f_height = 2
        val_result = self.unet.evaluate(model_path='U_BS2_I256_note.h5')
        self.assertNotEqual(val_result[0][1], 0, 'kappa is null')
        self.assertNotEqual(val_result[0][0], 0, 'loss is null')

    def test_kappa(self):
        a = np.array([[[0], [1]], [[1], [0]]], dtype=int)
        y_true = [tf.convert_to_tensor(a)]
        kappa = self.unet.kappa(y_true, y_true, False)
        self.assertEqual(kappa, 1)

    def test_prc_kappa(self):
        vstep = 2
        self.unet.load_data(4)  # a frame contain 4 framelets thus the height, weight should be 512*512
        self.unet.build()
        self.unet.train()
        model = tf.keras.models.load_model('U_BS2_I256_note.h5',
                                           custom_objects={'kappa': self.unet.kappa})
        self.unet.load_data(4)
        optthresh = self.unet.PRC_kappa(self.unet.test_dataset, vstep, model)
        self.assertNotEqual(optthresh, 0, 'optthresh should greater than 0')

    def test_wholepic(self):
        vstep = 2
        self.unet.load_data(4)  # a frame contain 4 framelets thus the height, weight should be 512*512
        self.unet.build()
        self.unet.train()
        model = tf.keras.models.load_model('U_BS2_I256_note.h5',
                                           custom_objects={'kappa': self.unet.kappa})
        self.unet.f_width = 2
        self.unet.f_height = 2
        compmsk2, [kap] = self.unet.wholepic(self.unet.val_ds, model, vstep, threshold=None)
        compmsk2.shape
        self.assertEqual(compmsk2.shape[0], 512)
        self.assertEqual(compmsk2.shape[1], 512)

    def test_test_model(self):
        self.unet.load_data(4)
        self.unet.build()
        self.unet.train()
        self.unet.f_width = 2
        self.unet.f_height = 2
        self.unet.opvalTH = 0.9998491  # for test convinience
        tresult = self.unet.test_model(model_path='U_BS2_I256_note.h5', n=4)
        self.assertIsNotNone(tresult)

    def test_combinetif(self):
        self.unet.load_data(4)
        self.unet.f_width = 2
        self.unet.f_height = 2
        dst = self.unet.combinetif(self.unet.images[0])
        self.assertEqual(dst.size, (512, 512))
        self.assertEqual(dst.im.bands, 4)

# def main():
#     test_suite = unittest.TestSuite()
#     test_suite.addTest(unittest.makeSuite(TestUnet))
#     test_suite.addTest(unittest.makeSuite(TestDataLoader))
#     test_suite.addTest(unittest.makeSuite(TestCodeFormat))
#     return test_suite
#
#
# if __name__ == "__main__":
#     my_suite = main()
#     runner = unittest.TextTestRunner()
#     runner.run(my_suite)
