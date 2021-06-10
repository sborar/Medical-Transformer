import numpy as np
import os
from os.path import join, isdir
from shutil import copyfile
from pathlib import Path


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
            root = join(body_part_folder, 'img', f)
            result = list(Path(root).rglob("*.png"))
            if np.random.random() < 0.7:
                [copyfile(str(r), join(train_data_folder, 'img', str(r).split('/')[-1])) for r in result]
                [copyfile(str(r), join(train_data_folder, 'labelcol', str(r).split('/')[-1])) for r in result]
            else:
                [copyfile(str(r), join(test_data_folder, 'img', str(r).split('/')[-1])) for r in result]
                [copyfile(str(r), join(test_data_folder, 'labelcol', str(r).split('/')[-1])) for r in result]


split_dataset(['Brain'])
