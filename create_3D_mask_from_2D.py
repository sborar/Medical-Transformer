# get file name
# get mask from the the place
# get all the images from that folder name and then stack them
# then put them to roi and then return

import glob
import gzip
import numpy as np
import os
import shutil
import sys
from PIL import Image
from numpy import asarray
from os.path import join

sys.path.append("../mvrepo/src/mvworker/code")

from mvml.utils import save_im
from mvdicom import Masks
from mvutils import save_pickle, load_pickle

# use the meta name to put things in a certain folder
base = "results"

scan_dir = "/azure-ml/mvinterns/deepmind-headneck-0/ct"

files = os.listdir(base)

scans = set([file.split('_')[0] for file in files])
output_masks = 'resulting_masks/'
output_videos = 'resulting_mask_videos/'
for scan in scans:
    image_files = glob.glob(join(base, scan + '*'))
    np_images = [asarray(Image.open(image_file).resize((512, 512))) for image_file in image_files]
    three_d_image = np.stack(np_images, axis=0)

    mask_dir = join(scan_dir, "masks_" + scan + "/")

    f = join(mask_dir, 'Brain.pkz')

    # checking if it is a file
    mask_data = np.load(gzip.open(f, 'rb'), allow_pickle=True)
    bool_mask = three_d_image != 255

    bbox = mask_data['bbox']
    b = [bbox[i] for i in [0, 3, 1, 4, 2, 5]]
    cropped_mask = bool_mask[b[0]: b[1], b[2]: b[3], b[4]: b[5]]
    mask_data['roi'] = cropped_mask
    mask_data['shape'] = (437,512,512)


    try:
        shutil.copytree(mask_dir, join(output_masks, 'masks_' + scan))
    except IOError as io_err:
        pass

    save_pickle(join(output_masks, 'masks_' + scan, 'Brain.pkz'), mask_data, True)

    im_file = join(scan_dir, 'image_' + scan + '.npz')
    # folder where videos will be saved
    im = np.load(im_file)['arr_0']
    masks = Masks(join(output_masks, 'masks_' + scan))
    masks_full_3d = masks.get_binary_masks()
    print(masks_full_3d.shape)  # 4D array: (31, 191, 512, 512)
    os.makedirs(os.path.dirname(output_videos), exist_ok=True)
    save_im(im, masks, dir_save=output_videos, name=scan + 'faster-saving-from-crops')
    save_im(im, masks_full_3d, dir_save=output_videos, name=scan + 'slower-saving-from-full-sized-masks')
