import glob
import gzip
import numpy as np
import os
import pandas as pd
from os.path import join, isdir

# use the meta name to put things in a certain folder
base = "/azure-ml/mvinterns/deepmind-headneck-0/ct/"

image_dataname = "image*"

raw_data_folder = "raw_dataset"

# get all files with pattern
image_files = glob.glob(join(base, image_dataname))

# for each file
img_data = pd.DataFrame(
    columns=['img', 'Bone_Mandible', 'SpinalCanal', 'Glnd_Lacrimal_L', 'Lung_R', 'Glnd_Submand_R', 'Glnd_Lacrimal_R',
             'Cochlea_L', 'OpticNrv_cnv_R', 'Lens_R', 'SpinalCord', 'Parotid_R', 'Glnd_Submand_L', 'Brainstem',
             'OpticNrv_cnv_L', 'Cochlea_R', 'Eye_R', 'Lens_L', 'Lung_L', 'Brain', 'Eye_L', 'Parotid_L'])
count = 0
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
        print('Saving ' + mask_data['name'])
        shape = mask_data['shape']  # shape of scan
        bbox = mask_data['bbox']
        cropped_mask = mask_data['roi']
        body_part_name = mask_data['name']
        
        x = image.shape[0]
        body_part_folder = join(raw_data_folder, body_part_name)
        if cropped_mask is None:
            print('roi is none')
            for i in range(x):
                img_name = join(body_part_folder, 'img', pid, scanid, pid_scanid + "_" + str(i) + ".png")
                img_data.loc[count, 'img'] = img_name
                img_data.loc[count, body_part_name] = -1
                count += 1
            continue

        mask = np.zeros(shape, dtype=np.bool)
        try:
            b = [bbox[i] for i in [0, 3, 1, 4, 2, 5]]  # get it in (z_min, z_max, y_min, y_max, x_min, x_max)
            mask[b[0]: b[1], b[2]: b[3], b[4]: b[5]] = cropped_mask
            z_min = bbox[0]
            z_max = bbox[3]

            mask_rgb = (mask[:, :, :] * 255).astype(np.uint8)
            pid_scanid_folder = join(body_part_folder, "img", pid, scanid)
            if not isdir(pid_scanid_folder):
                os.makedirs(join(body_part_folder, "img", pid, scanid))
                os.makedirs(join(body_part_folder, "labelcol", pid, scanid))

            # img_data.

            for i in range(x):
                img_name = join(body_part_folder, 'img', pid, scanid, pid_scanid + "_" + str(i) + ".png")
                img_data.loc[count, 'img'] = img_name
                if i < z_min or i > z_max:
                    img_data.loc[count, body_part_name] = -1
                else:
                    img_data.loc[count, body_part_name] = 1
                count += 1
        except Exception as e:
            print(e)
            continue

    print()

img_data = img_data.fillna(0)
img_data.to_csv('img_data.csv')
