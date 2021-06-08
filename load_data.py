import glob
import gzip
import numpy as np
import os
from PIL import Image
from os.path import join, isdir

image_folder = None
meta_folder = None
mask_folder = None

# use the meta name to put things in a certain folder
base = "/azure-ml/mvinterns/deepmind-headneck-0/ct"

image_dataname = "image*"

raw_data_folder = "raw_dataset"
# raw_data_folder = "~/sheetal_project/src/Medical-Transformer/raw_dataset"


# get all files with pattern
image_files = glob.glob(join(base, image_dataname))
# for each file

for image_file in image_files:
    try:
        pid_scanid = image_file.split('/')[-1][6:-4]
        print(pid_scanid)

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

        mask = np.zeros(shape, dtype=np.bool)
        try:
            b = [bbox[i] for i in [0, 3, 1, 4, 2, 5]]  # get it in (z_min, z_max, y_min, y_max, x_min, x_max)
            mask[b[0]: b[1], b[2]: b[3], b[4]: b[5]] = cropped_mask

            mask_rgb = (mask[:, :, :] * 255).astype(np.uint8)

            body_part_folder = join(raw_data_folder, body_part_name)
            if not isdir(body_part_folder):
                os.makedirs(join(body_part_folder, "img"))
                os.makedirs(join(body_part_folder, "labelcol"))

            x = image.shape[0]

            for i in range(x):
                output_image_fname = join(body_part_folder, 'img', pid_scanid + str(i) + ".png")
                output_label_fname = join(body_part_folder, 'labelcol', pid_scanid + str(i) + ".png")
                Image.fromarray(np.uint8(image[i, :, :])).convert('RGB').resize((128, 128)).save(
                    output_image_fname)
                Image.fromarray(np.uint8(mask_rgb[i, :, :])).convert('RGB').resize((128, 128)).save(
                    output_label_fname)
        except Exception as e:
            print(e)
            continue

    print()
