import glob
import gzip
import numpy as np
import os
import pandas as pd
from PIL import Image
from os.path import join, isdir

# use the meta name to put things in a certain folder
base = "/azure-ml/mvinterns/deepmind-headneck-0/ct/"

image_dataname = "image*"

raw_data_folder = "raw_dataset"

# get all files with pattern
image_files = glob.glob(join(base, image_dataname))

t = set(list([]))
for image_file in image_files:

    try:
        pid_scanid = image_file.split('/')[-1][6:-4]

        print(pid_scanid)
        pid = pid_scanid.split('.')[0]
        scanid = pid_scanid.split('-')[-1].split('_')[0]

        meta_dataname = join(base, "meta_" + pid_scanid + ".pkz")
        image = np.load(image_file, allow_pickle=True)["arr_0"]
        meta_data = np.load(gzip.open(meta_dataname, 'rb'), allow_pickle=True)

        masksname = join(base, "masks_" + pid_scanid + "/")
        masks = []

        for filename in os.listdir(masksname):
            f = join(masksname, filename)

            # checking if it is a file
            mask = np.load(gzip.open(f, 'rb'), allow_pickle=True)
            masks.append(mask)
    except Exception as e:
        print(e)
        continue

    if not masks:
        continue

    shape = meta_data['shape']  # scan shape
    spacing = (meta_data['ND_SliceSpacing'], meta_data['PixelSpacing'][1], meta_data['PixelSpacing'][0])
    
    for mask_data in masks:
        body_part_name = mask_data['name']
        t.add(body_part_name)
    
print(t)
