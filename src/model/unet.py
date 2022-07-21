import gc
import math

import cv2
import numpy as np
from PIL import Image as Image

from src.model.base_model import BaseModel
from src.data_loader.data_loader import DataLoader
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow as tf
from src.data_loader.preprocessing import Preprocess
import tensorflow.keras.backend as K
from sklearn.utils.extmath import stable_cumsum


class Unet(BaseModel):
    """Unet Model Class"""

    def __init__(self, config, fold):
        super().__init__(config)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
        self.model = None
        self.output_channels = self.config.model.output
        self.images = None
        self.masks = None
        self.test_images = None
        self.test_masks = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0
        self.fold = fold
        self.image_size = self.config.data.image_size
        self.train_dataset = []
        self.test_dataset = []
        self.val_ds = []
        self.folds = [0, 1, 2, 3, 4]
        self.preprocessing = None
        self.history = []
        self.val_result = []
        self.teresult = []
        self.teresult_p = []
        self.custom_TH = []
        self.opvalTH = None
        self.f_height = 45
        self.f_width = 48

    def load_data(self, n=2160):
        """Loads and Preprocess data """
        [self.images, self.masks] = DataLoader().load_data(self.config.data.images_path, self.config.data.masks_path, n)

        self.preprocessing = Preprocess(self.fold, self.batch_size, self.folds, self.image_size, self.images,
                                        self.masks, self.images, self.masks)

        self.train_dataset, self.test_dataset = self.preprocessing.create_trainset()
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.val_ds = self.preprocessing.create_valds()

    def _upsample(self, filters, size):
        return [layers.Conv2DTranspose(filters, size, strides=2, padding='same'), layers.BatchNormalization(),
                layers.ReLU()]

    def build(self):
        """ Builds the Keras model based """
        inputs = layers.Input(shape=(self.config.data.image_size, self.config.data.image_size, 3), name='input')
        # xx = layers.Dense(3,activation='sigmoid')(inputs)

        # >3 channels
        #     input1 = layers.Conv2D(channels,(3,3),padding='same')(inputs)
        #     input2 = layers.Conv2D(channels,(3,3),padding='same')(input1)
        #     input = input = layers.Conv2D(3,(1,1),padding='same')(input2)
        #     skips = base_model(input)

        # 3 channels
        # h = self.image_size

        base_model = tf.keras.applications.VGG16(input_shape=self.config.model.input, include_top=False)

        vgg16_layer_names = [
            'block1_conv2',  # 1
            'block2_conv2',  # 1/2
            'block3_conv3',  # 1/4
            'block4_conv3',  # 1/8
            'block5_conv3',  # 1/16
        ]
        vgg16_layer_names.reverse()
        vgglayers = base_model.outputs + [base_model.get_layer(name).output for name in vgg16_layer_names]

        base_model = tf.keras.Model(inputs=base_model.input, outputs=vgglayers)
        skips = base_model(inputs)

        x = skips[0]

        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        for layer in self._upsample(512, (3, 3)):
            x = layer(x)

        for ch, skip in zip([256, 128, 64, 32], skips[1:-1]):
            x = layers.Concatenate()([x, skip])
            for layer in self._upsample(ch, (3, 3)):
                x = layer(x)

        x = layers.Concatenate()([x, skips[-1]])
        x = layers.Conv2D(96, (1, 1), padding='same')(x)

        x = layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='softmax', name='output')(x)
        # x = layers.Dense(classes, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=x, name='unet')
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.custom_cost_function
        #     iou = tf.keras.metrics.IoU(num_classes=2, target_class_id=[1])
        tf.config.run_functions_eagerly(True)
        self.model.compile(loss=loss, optimizer=opt, metrics=['acc', self.kappa])
        print(self.model.summary())
        #
        # def custom_cost_function(self, y, y_pred):
        #     weight = 10.
        #     y = K.cast(K.squeeze(K.one_hot(K.cast(y, 'int64'), self.output_channels), -2), 'float32')
        #
        # return K.sum(y * (-1) * tf.convert_to_tensor([1., weight]) * tf.math.log(y_pred + 1e-9), -1)

    def train(self):
        """Compiles and trains the model"""
        self.validation_steps = math.ceil(len(self.images[0]) * 1 / self.batch_size)
        self.steps_per_epoch = math.ceil(len(self.images[0]) * 5 / self.batch_size)

        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.test_dataset)
        self.model.save('U_BS%d_I%d_%s.h5' % (self.batch_size, self.image_size, 'note'))

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self, model_path=None):
        """Predicts resuts for the test dataset"""

        vresult = []
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path,
                                                    custom_objects={'kappa': self.kappa})
        else:
            self.model.load_weights('../data/U_BS%d_I%d_%s.h5' % (self.batch_size, self.image_size, 'note'))
        self.opvalTH = self.PRC_kappa(self.test_dataset, self.validation_steps, self.model)
        self.custom_TH.append(self.opvalTH)
        tf.keras.backend.clear_session()
        valresult = self.model.evaluate(self.val_ds, steps=math.ceil(len(self.images[0]) / self.batch_size))
        out, met = self.wholepic(self.val_ds, self.model, math.ceil(len(self.images[0]) / self.batch_size),
                                 self.opvalTH)
        vresult.append([valresult[0]] + met)
        self.val_result.append(vresult)
        print(self.opvalTH, self.val_result)
        return vresult

    def test_model(self, model_path=None, n=2160):
        [self.test_images, self.test_masks] = DataLoader().load_data(self.config.data.test_images_path,
                                                                     self.config.data.test_masks_path, n)
        test = [x for x in range(len(self.test_images))]
        tresult = []
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path,
                                                    custom_objects={'kappa': self.kappa})
        else:
            self.model.load_weights('U_BS%d_I%d_%s.h5' % (self.batch_size, self.image_size, 'note'))
        # self.model.load_weights('U_BS64_I256_FINAL_5FC4.h5')
        for i in test:
            tds = self.preprocessing.create_testds(test, i)
            out, met = self.wholepic(tds, self.model, math.ceil(len(self.test_images[0]) / self.batch_size),
                                     self.opvalTH)
            allmask = self.combinemsk(self.test_masks[test[i]])
            close10 = cv2.morphologyEx(np.uint8(out), cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))  # Remove
            out_p10 = cv2.morphologyEx(np.uint8(close10), cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))  # Close
            pmet_10 = self.kappa(allmask, out_p10, False)
            tresult.append(pmet_10)
        # tresult.append()
        return np.average(tresult)

    def kappa(self, y_true, y_pred, argm=True):
        if argm:
            y_pred2 = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
        else:
            y_pred2 = tf.reshape(y_pred, [-1])
        y_true2 = tf.reshape(y_true, [-1])
        conf = tf.cast(tf.math.confusion_matrix(y_true2, y_pred2, num_classes=self.output_channels), tf.float32)  # 2
        actual_ratings_hist = tf.reduce_sum(conf, axis=1)
        pred_ratings_hist = tf.reduce_sum(conf, axis=0)

        # print(conf,actual_ratings_hist,pred_ratings_hist)
        nb_ratings = tf.shape(conf)[0]
        weight_mtx = tf.zeros([nb_ratings, nb_ratings], dtype=tf.float32)
        diagonal = tf.ones([nb_ratings], dtype=tf.float32)
        weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
        gc = actual_ratings_hist * pred_ratings_hist
        conf = tf.cast(conf, tf.float32)
        totaln = tf.cast(tf.shape(y_true2)[0], tf.float32)
        up = tf.cast(totaln * tf.reduce_sum(conf * weight_mtx), tf.float32) - tf.cast(tf.reduce_sum(gc), tf.float32)
        down = tf.cast(totaln ** 2 - tf.reduce_sum(gc), tf.float32)
        # print(weight_mtx,gc,conf,up,down)

        if tf.math.is_nan(up / down):
            return 0.
        return up / down

    def PRC_kappa(self, tds, vstep, model):
        y_trueo = []
        for _, m in tds.take(vstep):
            try:
                y_trueo = np.concatenate((y_trueo, m.numpy()))
            except Exception:
                y_trueo = m.numpy()
        wout = model.predict(tds, verbose=0, steps=vstep)
        y_true = y_trueo.ravel()
        y_score = wout[:, :, :, 1].ravel()
        del wout
        del y_trueo

        gc.collect()

        y_true = (y_true == 1)
        desc_score_indices = np.argsort(y_score)[::-1]

        np.take(y_score, desc_score_indices, out=y_score)
        np.take(y_true, desc_score_indices, out=y_true)
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        del distinct_value_indices
        del desc_score_indices

        gc.collect()
        y_true2 = y_true[::-1]
        del y_true
        tns = stable_cumsum(np.invert(y_true2))[::-1][threshold_idxs]
        # fns = stable_cumsum(1-np.invert(y_true2))[::-1][threshold_idxs]
        total = tps[-1] + tps[0] + fps[0] + tns[0]
        fns = total - fps - tns - tps
        del y_true2
        del total

        gc.collect()
        total = (tns + tps + fns + fps)[0]
        po = (tns + tps) / total
        p1 = ((tps + fps) / total) * ((tps + fns) / total)
        p2 = ((tns + fps) / total) * ((tns + fns) / total)
        pe = p1 + p2
        kap = (po - pe) / (1 - pe)

        optthresh = y_score[threshold_idxs][np.argmax(kap)]
        print('Best Kappa: %.3f  when threshold = %.3f ' % (max(kap), optthresh))

        return optthresh

    def wholepic(self, w_ds, model, vstep, threshold=None):
        allm = []
        # y_true = []
        for _, m in w_ds.take(vstep):
            # try:
            #   y_true = np.concatenate((y_true,m.numpy()))
            # except:
            #   y_true = m.numpy()
            for r in m:
                allm.append(r)
        wout = model.predict(w_ds, verbose=0, steps=vstep)
        if threshold is None:
            w_output = tf.argmax(wout, axis=3)
        else:
            w_output = wout[:, :, :, 1] > threshold

        # CALCULATE THE SEGMENTATION RESULT MATRICS AFTER COMBINE
        w_output = tf.cast(w_output, tf.int8)
        # ytflat = y_true.flatten().astype('int8')
        # woflat = w_output.numpy().flatten()
        # print(pd.crosstab(ytflat, woflat, rownames = ['label'], colnames = ['predict'])) #CONFUSION MATRIX
        w_output = tf.expand_dims(w_output, -1)
        # los = keras.losses.sparse_categorical_crossentropy(allm,wout)
        kap = self.kappa(allm, w_output, False).numpy()

        del wout, allm

        co = 0
        for i in range(
                self.f_height):
            for r in range(self.f_width):  # why 48?
                mskn = w_output[co]
                mskn = tf.squeeze(mskn, -1)
                co += 1
                if r == 0:
                    temp = mskn
                else:
                    temp = np.concatenate((temp, mskn), axis=0)
            if i == 0:
                compmsk2 = temp
            else:
                compmsk2 = np.concatenate((compmsk2, temp), axis=1)

        # return compmsk2,[kap, accuracy_score(ytflat,woflat), iou0, iou1, pre,rec]
        return compmsk2, [kap]

    def combinemsk(self, whole_mskl):
        # only for showing
        co = 0
        for i in range(self.f_height):
            for r in range(self.f_width):
                msk = tf.io.read_file(whole_mskl[co])
                msk = tf.image.decode_jpeg(msk, channels=1)
                msk = tf.image.convert_image_dtype(msk, tf.float32)
                msk = tf.math.logical_and(msk < 256, msk > 0)
                mskn = msk.numpy().squeeze()
                if r == 0:
                    temp = mskn
                else:
                    temp = np.concatenate((temp, mskn), axis=0)
                co += 1
            if i == 0:
                compmsk = temp
            else:
                compmsk = np.concatenate((compmsk, temp), axis=1)

        return compmsk

    def combinetif(self, whole_imgl):
        # only for showing
        dst = Image.new('RGBX', (256 * self.f_height, 256 * self.f_width))
        co = 0
        for i in range(self.f_height):
            temp = Image.new('RGBX', (256, 256 * self.f_width))
            for r in range(self.f_width):
                img2 = Image.open(whole_imgl[co])
                co += 1
                temp.paste(img2, (0, 256 * r))
            dst.paste(temp, (i * 256, 0))
        return dst
