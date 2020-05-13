import os
import numpy as np
import cv2

from create_poly_mask import create_poly_mask

input_vec_dir = "../../data/train/AOI_11_Rotterdam/geojson_buildings/"
input_img_dir = "../../data/train/AOI_11_Rotterdam/PS-RGB/"
#input_sar_dir = "../../data/train/AOI_11_Rotterdam/SAR-Intensity/"
output_dir = "../../data/train/AOI_11_Rotterdam/gt_masks/"

for fi in os.listdir(input_vec_dir):
    if fi.endswith(".geojson"):
        input_vec_file = os.path.join(input_vec_dir,fi)
        input_img_file = os.path.join(input_img_dir,fi.replace('_Buildings_', '_PS-RGB_').replace('.geojson', '.tif'))
        output_file = os.path.join(output_dir,fi.replace('_Buildings_', '_PS-RGB_').replace('.geojson', '.tif'))
        # create_poly_mask need to be checked
        mask = create_poly_mask(input_img_file, input_vec_file, npDistFileName=output_file, noDataValue=0, burn_values=255)