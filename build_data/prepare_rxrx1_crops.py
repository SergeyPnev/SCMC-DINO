import os
import ast
import cv2
import skimage
import datetime
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
import albumentations as A

root_dir = "/raid/user/data/RXRX1/rxrx1/images/"
save_dir = "/raid/user/data/RXRX1/rxrx1/crops/"
df = pd.read_pickle("/raid/user/data/RXRX1/rxrx1/metadata.pkl")
w = 12

def get_image(root_dir, experiment, plate, site, well):
    full = np.zeros((512, 512, 6))
    for i in range(6):
        full[:, :, i] = skimage.io.imread(root_dir + f"{experiment}/Plate{plate}/{well}_s{site}_w{i+1}.png")
    return full

print("df shape: ", df.shape)
start = datetime.datetime.now()
for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(idx, datetime.datetime.now() - start)
        start = datetime.datetime.now()
    experiment = row["Metadata_Experiment"]
    plate = row["Metadata_Plate"]
    site = row["Metadata_Site"]
    well = row["Metadata_Well"]

    example = get_image(root_dir, experiment, plate, site, well)

    crop_coords = row["crops"]
    folder_dir = save_dir + f"{experiment}/Plate{plate}"

    os.makedirs(folder_dir, exist_ok=True)
    for x,y in crop_coords:
        if (x - w > 0) and (x + w < 512) and (y - w > 0) and (y + w < 512):
            crop = example[y-w:y+w, x-w:x+w , ...].astype(np.uint8)
            tif.imwrite(folder_dir + f"/{well}_s{site}_{y}_{x}.tiff", crop, photometric='minisblack')
