from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import cv2
import gc

import random, cv2, os, sys, shutil

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

import time


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def crop_image(imgs, out_path):
    imgs = glob.glob(imgs + '/*.png').sort()[:100]
    print('/nCrop Image')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    width = 11520
    height = 12288
    crop_width = 256
    crop_height = 256
    file_count = 1
    for file in imgs:
        # read image and resize
        ic = 1
        im = Image.open(file)
        im = im.resize((width, height), Image.ANTIALIAS)

        # get file name and make a folder for it
        filename = os.path.basename(file).split('.')[0]

        print("crop ", file_count, ": ", file, "...")
        for i in range(0, width, crop_width):
            for r in range(0, height, crop_height):
                a = im.crop((i, r, i + crop_width, r + crop_height))
                a.save(out_path + r'/{}_{}.png'.format(ic, filename))
                ic = ic + 1
        file_count = file_count + 1


def image_clustering(image_paths, n_clusters, output_dir,out_crop_mask,out_crop_img):
    # if folder exist it mean it come from previous fold -> remove create a new one



    # SAVE THE FRAMELET IMAGES WITH RICE
    rice_path = output_dir + "/clusterX"
    ricegt_path = output_dir + "/clusterGTX"

    for i in range(n_clusters):
        if not os.path.exists(output_dir + "/cluster" + str(i)):
            os.makedirs(output_dir + "/cluster" + str(i))
        if not os.path.exists(output_dir + "/clusterGT" + str(i)):
            os.makedirs(output_dir + "/clusterGT" + str(i))
    #         os.makedirs(output_dir+"/H" + str(i))
    print("/n Object of class /image_clustering/ has been initialized.")

    images = []
    gt = []
    for image in image_paths:
        img_name = image.split('/')[-1]
        gt_name = img_name.replace("_mtbn", "")
        gt = cv2.imread(out_crop_mask+"/" + gt_name)
#         print(np.unique(gt))
        # CHECK THE FRAMELET WITH RICE
        if (255 in gt) == True:
            #             images.append(img)
            # THE FRAMELET WITH RICE WILL SAVE HERE FOR CLUSTERING USING K-MEANS
            shutil.copy2(out_crop_img+"/" + img_name , output_dir + "/clusterX")
            shutil.copy2(out_crop_mask+"/" + gt_name, output_dir + "/clusterGTX")
            images.append(cv2.imread(image))
            continue
        else:
            shutil.copy2(out_crop_img+"/" + img_name , output_dir + "/cluster2")
            shutil.copy2(out_crop_mask+"/" + gt_name, output_dir + "/clusterGT2")
            continue

    # paths = os.listdir(rice_path)
    paths = os.listdir(rice_path)
    max_examples = len(paths)
    rice_name = paths[:max_examples]

    number = len(images)
    print(number)
    images = np.float32(images).reshape(len(images), -1)
    print("Images Value of Rice: ", images)
    images /= 255

    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(images)
    predictions = model.predict(images)

    plt.scatter(images[:, 0], images[:, 1], c=predictions, s=50, cmap='viridis');

    for i in range(max_examples):
        shutil.copy2(rice_path + "/" + rice_name[i], output_dir + "/cluster" + str(predictions[i]))
        shutil.copy2(ricegt_path + "/" + rice_name[i].replace("_mtbn", ""), output_dir + "/clusterGT" + str(predictions[i]))

    print("/n Clustering complete! /n/n Clusters and the respective images are stored in the /output/ folder.")


def move(imgs, masks, no_img, no_mask):
    imgfiles = os.listdir(imgs)
    rand = random.sample(imgfiles, 100)  # 2160

    for filename in rand:
        shutil.move(os.path.join(imgs, filename), no_img)
        shutil.move(os.path.join(masks, filename), no_mask)


def moveback(imgs, masks, no_img, no_mask):
    imgfiles = os.listdir(no_img)

    for filename in imgfiles:
        shutil.move(os.path.join(no_img, filename), imgs)
        shutil.move(os.path.join(no_mask, filename), masks)

    print("Done")


def filename2(f1, imgs, masks):
    os.getcwd()
    collection = imgs
    for i, filename in enumerate(os.listdir(collection)):
        #         print(filename)

        f1 = f1

        old_img = imgs + "/" + filename
        old_mask = masks + "/" + filename
        new_img = imgs + "/" + str(i + 1) + '_' + f1 + ".png"
        new_mask = masks + "/" + str(i + 1) + '_' + f1 + ".png"

        os.rename(old_img, new_img)
        os.rename(old_mask, new_mask)


def change_label(imgs):
    imgs = glob.glob(imgs + '/*.png')
    print('Change Label')
    i = 1
    for img in imgs:
        print('{}:{}'.format(i, img))
        im = np.array(Image.open(img))
        im = np.where(im >= 1, 1, im)
        #         im = np.where(im==1, 255, im)
        im = Image.fromarray(im)
        im.save(img)
        i = i + 1


def post(fold):
    if os.path.exists( "output_cluster/clusterX"):
        deletion("output_cluster/clusterX")
        deletion("output_cluster/clusterGTX")
        deletion( "output_cluster/cluster0")
        deletion( "output_cluster/clusterGT0")
        deletion( "output_cluster/cluster1")
        deletion( "output_cluster/clusterGT1")
        deletion( "output_cluster/cluster2")
        deletion( "output_cluster/clusterGT2")
        'FOLD'+str(fold)+'_3CLUSTER/F1C1_IMG_3C'
    if os.path.exists("clustered/FOLD"+str(fold)+"_3CLUSTER/img1"):
        deletion(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img0")
        deletion(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img1")
        deletion(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img2")
        deletion(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img3")
        deletion(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img4")

def deletion(folderpath):
    for x in glob.glob(folderpath+'/*.png'):
        os.remove(x)


def pre(fold):
    if not os.path.exists( "output_cluster/clusterX"):
        os.makedirs( "output_cluster/clusterX")
        os.makedirs( "output_cluster/clusterGTX")
        os.makedirs( "output_cluster/cluster2")
        os.makedirs( "output_cluster/clusterGT2")
    if not os.path.exists(r"clustered/FOLD"+str(fold)+"_3CLUSTER/img1"):
        os.makedirs("clustered/FOLD"+str(fold)+"_3CLUSTER/img0")
        os.makedirs("clustered/FOLD"+str(fold)+"_3CLUSTER/img1")
        os.makedirs("clustered/FOLD"+str(fold)+"_3CLUSTER/img2")
        os.makedirs("clustered/FOLD"+str(fold)+"_3CLUSTER/img3")
        os.makedirs("clustered/FOLD"+str(fold)+"_3CLUSTER/img4")
    if not os.path.exists(r"FOLD"+str(fold)+"_3CLUSTER/IMG"):
        os.makedirs("FOLD"+str(fold)+"_3CLUSTER/IMG")
    if not os.path.exists(r"FOLD"+str(fold)+"_3CLUSTER/MASK"):
        os.makedirs("FOLD"+str(fold)+"_3CLUSTER/MASK")
    if not os.path.exists(r"FOLD"+str(fold)+"_3CLUSTER/F1C1_IMG_3C"):
        os.makedirs(r"FOLD"+str(fold)+"_3CLUSTER/F1C1_IMG_3C")
    if not os.path.exists(r"FOLD"+str(fold)+"_3CLUSTER/F1C1_MASK_3C"):
        os.makedirs(r"FOLD"+str(fold)+"_3CLUSTER/F1C1_MASK_3C")

def main(fold):
    # CROP the image to 256 by 256
    for x in frames:
        pre(fold)
        print(x)
        frame = np.loadtxt(x, str)


        print("/n /t START /n")
        image_paths = [str for str in cropped_images if
                       any(sub in str for sub in frame[:-3])]
        tic()
        image_clustering(image_paths, number_of_clusters, output_dir,out_crop_mask,out_crop_img)
        tac()
        img = 'FOLD'+str(fold)+'_3CLUSTER/F1C1_IMG_3C'
        mask = 'FOLD'+str(fold)+'_3CLUSTER/F1C1_MASK_3C'

        l_imgfiles0 = 'output_cluster/cluster0'
        l_mskfiles0 = 'output_cluster/clusterGT0'
        l_imgfiles1 = 'output_cluster/cluster1'
        l_mskfiles1 = 'output_cluster/clusterGT1'
        l_imgfiles2 = 'output_cluster/cluster2'
        l_mskfiles2 = 'output_cluster/clusterGT2'

        imgfiles0 = os.listdir('output_cluster/cluster0')
        mskfiles0 = os.listdir('output_cluster/clusterGT0')
        imgfiles1 = os.listdir('output_cluster/cluster1')
        mskfiles1 = os.listdir('output_cluster/clusterGT1')
        imgfiles2 = os.listdir('output_cluster/cluster2')
        mskfiles2 = os.listdir('output_cluster/clusterGT2')

        # C1 = 2280
        # C2 = 3600
        # C3 = 4320
        C4 = 3600

        r_imgfiles0 = random.sample(imgfiles0, C4)

        for filename in r_imgfiles0:
            shutil.copy(os.path.join(l_imgfiles0, filename), img)
            shutil.copy(os.path.join(l_mskfiles0, filename.replace("_mtbn", "")), mask)
        print(len(glob.glob(img+"/*.png")))
        print(len(glob.glob(mask+"/*.png")))

        r_imgfiles1 = random.sample(imgfiles1, C4)
        for filename in r_imgfiles1:
            shutil.copy(os.path.join(l_imgfiles1, filename), img)
            shutil.copy(os.path.join(l_mskfiles1, filename.replace("_mtbn", "")), mask)
        print(len(glob.glob(img+"/*.png")))
        print(len(glob.glob(mask+"/*.png")))

        r_imgfiles2 = random.sample(imgfiles2, C4)
        for filename in r_imgfiles2:
            shutil.copy(os.path.join(l_imgfiles2, filename), img)
            shutil.copy(os.path.join(l_mskfiles2, filename.replace("_mtbn", "")), mask)
        print(len(glob.glob(img+"/*.png")))
        print(len(glob.glob(mask+"/*.png")))

        # CHANGE THE FOLDER DESTINATION:
        # COMBINATION 1 RANGE(1,5)
        # COMBINATION 2 RANGE(1,6)
        # COMBINATION 3 RANGE(1,7)
        n_img = r"clustered/FOLD"+str(fold)+"_3CLUSTER/img"+str(i)
        n_mask = r"clustered/FOLD"+str(fold)+"_3CLUSTER/mask"+str(i)
        for i in range(0,5):# 5 round, for 4 training and 1 validation


            #CHANGE THE IMAGE NAME, 11 for img1, 22 for img2, 33 for img3, or 44 for img4
            f1 = str(i)+str(i)
            print(img, mask, n_img, n_mask)

            #RUN TO MOVE THE DATASET 
            move(img, mask, n_img, n_mask)

        #     RUN TO RENAME THE DATASET NAME
            filename2(f1, n_img, n_mask)
        for i in range(0,5):
            # MOVE BACK THE READY DATASET
            moveback(img, mask, n_img, n_mask)

        change_label(mask)
        fold += 1
        post(output_dir)



# out_crop_img = r"cropped_img"
# out_crop_mask = r"cropped_GT"
out_crop_img = r"/media/kuro/Data/PhD/COA/RICE/cropped_img"
out_crop_mask = r"/media/kuro/Data/PhD/COA/RICE/cropped_mask"
# crop_image(r"img/", out_crop_img)
# crop_image("mask/", out_crop_mask)

output_dir = 'output_cluster'
cropped_images = glob.glob(out_crop_img+'/*.png')
cropped_images.sort()
cropped_masks = glob.glob(out_crop_mask+'/*.png').sort()
number_of_clusters = 2

frames = glob.glob('*.txt')
print(frames)
frames.sort()


# -
if  __name__== "__main__":
    post(1)
    main(1)




