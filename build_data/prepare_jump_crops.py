import os
import ast
import cv2
import random
import skimage
import numpy as np
import pandas as pd
import tifffile as tiff
import albumentations as A
from datetime import datetime
import matplotlib.pyplot as plt
join = os.path.join

root_dir = "/raid/user/data/JUMP/cpg0004-lincs/inputs/"
save_dir = "/raid/user/data/JUMP/cpg0004-lincs/inputs_sc_tiff"
path = "/raid/user/data/JUMP/cpg0004-lincs/metadata/df_withDMSO_withCrops.csv"
df = pd.read_csv(path)


def transform_to_list(val):
    return ast.literal_eval(val)


def get_image(root_dir, img_paths):
    full = np.zeros((1080, 1080, 5))
    for i, path in enumerate(img_paths.split(",")):
        full[:, :, i] = skimage.io.imread(root_dir + path)
    return full


def crop_single_cell(image, x, y, scale=1, crop_size=40):
    half_crop = crop_size // 2
    x *= scale
    y *= scale
    x = int(x)
    y = int(y)
    img_height, img_width, img_depth = image.shape

    x_min = max(0, x - half_crop)
    x_max = min(img_height, x + half_crop)
    y_min = max(0, y - half_crop)
    y_max = min(img_width, y + half_crop)

    cropped_img = image[x_min:x_max, y_min:y_max, :]

    if cropped_img.shape[0] != crop_size or cropped_img.shape[1] != crop_size:
        cropped_img = np.pad(cropped_img,
                             ((0, crop_size - cropped_img.shape[0]),
                              (0, crop_size - cropped_img.shape[1]),
                              (0, 0)),
                             mode='constant')
    return cropped_img


start = datetime.now()
resize = A.Resize(96, 96)
for idx, row in df.iterrows():
    paths = df.iloc[idx]["combined_paths"]
    img = get_image(root_dir, paths)
    crops_xy = transform_to_list(df.iloc[idx]["crops"])

    crops = [resize(image=crop_single_cell(img, x, y))["image"] for x, y in crops_xy]

    for i, xy in enumerate(crops_xy):
        x, y = xy
        path = paths.split(",")[0]
        plate = path.split("/")[0]
        plate_dir = f"{save_dir}/{plate}"
        os.makedirs(plate_dir, exist_ok=True)

        # for j, path in enumerate(paths.split(",")):
        #     new_path = f"{path[:-4]}_{x}_{y}.png"
        #     # cv2.imwrite(f"{save_dir}/{new_path}", crops[i][..., j])

        path = paths.split(",")[0]
        new_path = f"{path[:-4]}_{x}_{y}.tiff"
        tiff.imwrite(f"{save_dir}/{new_path}", crops[i].astype(np.uint8))

    if idx % 100 == 0:
        print(f"{idx} out of {df.shape[0]} took {datetime.now() - start}")
        start = datetime.now()
