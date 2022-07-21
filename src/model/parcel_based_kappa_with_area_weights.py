import numpy as np
from PIL import Image
import glob
import os
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops
from sklearn.metrics import cohen_kappa_score
from skimage.transform import resize


def change_pixel_value_of_region(image_array, coordinates, target_value):
    """
    image_array:
        whole image
    coordinates:
        region.coords
    """
    for coord in coordinates:
        # print("coord:{}".format(coord))
        image_array[coord[0], coord[1]] = target_value

    return image_array


def get_pixel_counts_of_target_region_in_model_result(model_result_image_array, coordinates, target_value):
    """

    image_array:
        whole image
    coordinates:
        region.coords

    """
    count = 0
    for coord in coordinates:
        if model_result_image_array[coord[0], coord[1]] == target_value:
            # print("coord:{}".format(coord))
            count += 1

    return count


def calculate_parcel_based_kappa_with_area_weights(parcel_mask_path, pixel_based_model_result_path, ori_height,
                                                   ori_width, ratio_threshold=0.5):
    """
    ratio_threshold:
        On the pixel-based model result,
        if the pixel ratio judged as rice in one parcel is more than ratio_threshold,
        then that parcel will be regarded as rice in the parcel-based model result.
    """
    parcel_mask = np.array(Image.open(parcel_mask_path).convert("L"))
    pixel_based_model_result = np.array(Image.open(pixel_based_model_result_path).convert("L"))

    pixel_val = np.unique(pixel_based_model_result)

    parcel_based_model_result = np.zeros(parcel_mask.shape)

    # 像素"相鄰"且"具有相同值"才視為連通
    labeled_component_GT_mask, labeled_component_count = label(
        parcel_mask, background=0, return_num=True, connectivity=1)

    region_list = regionprops(labeled_component_GT_mask)
    for region in region_list:  # for each parcel region
        target_parcel_region = region

        pixel_judged_as_rice_count = get_pixel_counts_of_target_region_in_model_result(
            model_result_image_array=pixel_based_model_result,
            coordinates=target_parcel_region.coords,
            target_value=255)

        ratio_of_pixel_judged_as_rice = pixel_judged_as_rice_count / region.area
        # region.area: total pixels count in this region

        if ratio_of_pixel_judged_as_rice >= ratio_threshold:
            # regard all pixels of this target_parcel as being judged as rice(pixel value is 1)
            parcel_based_model_result = change_pixel_value_of_region(parcel_based_model_result,
                                                                     target_parcel_region.coords, 1)

    loc_parcel = np.where(parcel_mask != 0)  # the location information of all farmland parcel

    # Change the pixel value of parcel_mask before calculating kappa
    # make sure the meanning of pixel values are the same to parcel_mask and parcel-based model result
    # => non-rice pixel(background):0, rice pixel:1
    parcel_mask[np.where(parcel_mask == 1)] = 0  # regard the non-rice parcel as background on parcel_mask (GT)
    parcel_mask[np.where(parcel_mask == 2)] = 1  # change the pixel value in rice parcel to 1 on parcel_mask (GT)

    # ignore the pixels out of farmland parcel
    parcel_mask = parcel_mask[loc_parcel]
    parcel_based_model_result = parcel_based_model_result[loc_parcel]

    parcel_mask = parcel_mask.flatten()
    parcel_based_model_result = parcel_based_model_result.flatten()
    parcel_based_kappa_with_area_weight = cohen_kappa_score(parcel_mask, parcel_based_model_result)

    c_matrix = confusion_matrix(parcel_mask, parcel_based_model_result)
    print("confusion_matrix:\n", c_matrix)

    c_matrix_sum = c_matrix.sum()
    print("Ratio:")
    print(" {}  {}".format(round(c_matrix[0, 0] / c_matrix_sum, 5), round(c_matrix[0, 1] / c_matrix_sum, 5)))
    print(" {}  {}".format(round(c_matrix[1, 0] / c_matrix_sum, 5), round(c_matrix[1, 1] / c_matrix_sum, 5)))

    return parcel_based_kappa_with_area_weight
