import numpy as np
import os
from os.path import join, isdir
from shutil import copyfile

raw_data_folder = "raw_dataset"

train_data_folder = "train_dataset"
test_data_folder = "test_dataset"


def split_dataset(structures):
    if not isdir(train_data_folder):
        os.makedirs(join(train_data_folder, "img"))
        os.makedirs(join(train_data_folder, "labelcol"))

    if not isdir(test_data_folder):
        os.makedirs(join(test_data_folder, "img"))
        os.makedirs(join(test_data_folder, "labelcol"))

    for structure in structures:
        body_part_folder = join(raw_data_folder, structure)
        files = os.listdir(join(body_part_folder, 'img'))
        for f in files:
            if np.random.random() < 0.7:
                copyfile(join(body_part_folder, 'img', f), join(train_data_folder, 'img', f))
                copyfile(join(body_part_folder, 'labelcol', f), join(train_data_folder, 'labelcol', f))
            else:
                copyfile(join(body_part_folder, 'img', f), join(test_data_folder, 'img', f))
                copyfile(join(body_part_folder, 'labelcol', f), join(test_data_folder, 'labelcol', f))


split_dataset(['Brain'])
