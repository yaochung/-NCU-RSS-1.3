import random as rd
import tensorflow as tf


class Preprocess:
    def __init__(self, fold, batch_size, folds, image_shape, images, masks, test_images, test_masks):
        self.fold = fold
        self.batch_size = batch_size
        self.folds = folds
        self.images = images
        self.masks = masks
        self.image_shape = (image_shape, image_shape, 3)
        self.images = images
        self.masks = masks
        self.test_images = test_images
        self.test_masks = test_masks
        self.vals = [x for x in range(5) if x % 5 == self.fold]
        self.trilist = [x for r in range(len(self.images)) if r in self.folds for x in self.images[r] if x]
        self.trmlist = [x for r in range(len(self.masks)) if r in self.folds for x in self.masks[r] if x]
        self.valilist = [x for r in range(len(self.images)) if r in self.vals for x in self.images[r] if
                         x]  # sampling from images[vals]
        self.valmlist = [x for r in range(len(self.masks)) if r in self.vals for x in self.masks[r] if x]

    def create_trainset(self, ):
        tr_ds = tf.data.Dataset.from_generator(self.imggen, (tf.float32, tf.float32),
                                               args=[self.trilist, self.trmlist, len(self.trmlist), self.image_shape,
                                                     True])
        tr = tr_ds.map(self.fixx, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(self.batch_size)

        val_ds = tf.data.Dataset.from_generator(self.imggen, (tf.float32, tf.float32),
                                                args=[self.valilist, self.valmlist, len(self.valmlist),
                                                      self.image_shape, False])
        val = val_ds.map(self.fixx).batch(self.batch_size)
        return tr, val

    def imggen(self, imgpath, mskpath, stop, imgs, aug=False, whole=False):
        i = 0
        if whole:
            i = 1
        while i < stop:
            img = tf.io.read_file(imgpath[i])
            img = tf.image.decode_jpeg(img, channels=3)

            img = tf.image.convert_image_dtype(img, tf.float32)
            msk = tf.io.read_file(mskpath[i])
            msk = tf.image.decode_jpeg(msk, channels=1)
            msk = tf.image.convert_image_dtype(msk, tf.float32)
            msk = tf.math.multiply(msk, 255)
            msk = tf.cast(msk, tf.float32)
            if aug:
                if rd.random() > 0.5:
                    img = tf.image.flip_left_right(img)
                    msk = tf.image.flip_left_right(msk)
                if rd.random() > 0.5:
                    img = tf.image.flip_up_down(img)
                    msk = tf.image.flip_up_down(msk)
                if rd.random() > 0.6:
                    img = tf.image.random_brightness(img, 0.3)

            img = tf.image.per_image_standardization(img)
            # if CH_N<4:
            # img = tf.slice(img,[0,0,0],[-1,-1,CH_N])

            i += 1
            yield img, msk

    def fixx(self, img, msk):
        # image_shape : list
        # mask : the mask shape
        img.set_shape(self.image_shape)
        msk.set_shape([self.image_shape[0], self.image_shape[0], 1])
        return img, msk

    def create_valds(self):
        # 0, I dn't know why ida use j in range(1) which return 0
        vali = self.images[self.vals[0]]
        valm = self.masks[self.vals[0]]
        vali = [x for x in vali if x]
        valm = [x for x in valm if x]
        test_ds = tf.data.Dataset.from_generator(self.imggen, (tf.float32, tf.float32),
                                                 args=[vali, valm,
                                                       len(vali),
                                                       self.image_shape, False])
        testing = test_ds.map(self.fixx).batch(self.batch_size)
        return testing

    def create_testds(self, test, indx):
        # 0, I dn't know why ida use j in range(1) which return 0
        vali = self.test_images[test[indx]]
        valm = self.test_masks[test[indx]]
        vali = [x for x in vali if x]
        valm = [x for x in valm if x]
        test_ds = tf.data.Dataset.from_generator(self.imggen, (tf.float32, tf.float32),
                                                 args=[vali, valm,
                                                       len(vali),
                                                       self.image_shape, False])
        testing = test_ds.map(self.fixx).batch(self.batch_size)
        return testing
