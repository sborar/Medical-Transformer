import os
import numpy as np
import torch

from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print(image.shape)
        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)

        mask[mask<=127] = 0
        mask[mask>127] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # print(image.shape)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:
        
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, transform: Callable = None):

        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))

        # image = np.transpose(image,(2,0,1))

        image = correct_dims(image)

        image = self.transform(image)

        # image = np.swapaxes(image,2,0)

        return image, image_filename

def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:        
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}

# def save_im(im: np.ndarray,
#             pred: Union[np.ndarray, Masks] = None,
#             gt: Union[np.ndarray, Masks] = None,
#             clip_vals: Union[None, Tuple[float, float]] = (-200, 200),
#             dir_save: str = None,
#             name: str = 'name',
#             colors: np.ndarray = None,
#             quality: int = 6,
#             line_width: int = 1,
#             edge_filter: str = 'erosion',
#             frame_rate: int = 10,
#             save_avi: bool = True,
#             save_jpg: bool = False,
#             square: bool = True,
#             min_sz: int = 512,
#             norm: callable = norm_minmax,
#             do_copy: bool = True,
#             ):
#     # todo: plot contours in Z-direction
#     # todo: move through orthogonal views
#     # gt_ col: green [0,1,0] for all organs
#     # pred col: different for each organ
#     # if not compressed:
#     #     quality = 10  # 10 is for uncompressed videos
#     if do_copy:
#         im = im.copy()
#
#     if len(im.shape) == 4 and im.shape[-1] != 3:
#         im = np.squeeze(im, axis=0)
#     elif len(im.shape) not in [2, 3, 4]:
#         raise Exception(f'image should be either 2D or 3D but its shape is: {im.shape}')
#
#     if isinstance(clip_vals, (tuple, list)):
#         im = np.clip(im, clip_vals[0], clip_vals[1])
#
#     im = norm(im)
#     im = (255 * im).round().astype(np.uint8)
#
#     if dir_save is None:
#         dir_save = getcwd()
#
#     sz = np.array([im.shape[1], im.shape[2]])
#     n_slices = im.shape[0]
#
#     scale = 1 if min_sz is None else max(min_sz / sz)
#
#     if pred is not None:
#         n_cls = pred.shape[0] if isinstance(pred, np.ndarray) else len(pred.names2idx)
#     elif gt is not None:
#         n_cls = gt.shape[0] if isinstance(gt, np.ndarray) else len(gt.names2idx)
#     else:
#         n_cls = 1
#
#     if colors is None:
#         colors = get_colors(n_cls)
#
#     line_width_gt = line_width + 0
#     if edge_filter == 'roberts':
#         edge_filter_fun = roberts
#     elif edge_filter == 'sobel':
#         edge_filter_fun = sobel
#     elif edge_filter == 'dilation':  # dilation can often exceed the tight bbox, so do not use it with tightly cropped ROIs
#         def edge_filter_fun(im_):
#             ime = binary_dilation(im_)
#             return np.logical_xor(im_, ime)
#     elif edge_filter == 'erosion':  # erosion can sometimes delete the whole region
#         def edge_filter_fun(im_):
#             ime = binary_erosion(im_)
#             return np.logical_xor(im_, ime)
#     else:
#         raise NotImplementedError(edge_filter)
#
#     video = __draw_contours_3d__(im, pred, gt, scale, colors, edge_filter_fun, line_width, line_width_gt, 0.5)
#     del im
#
#     video = np.stack(video, axis=0)
#     video = np.flip(video, axis=3)  # bgr -> rgb
#     if save_avi:
#         if square:
#             video, new_sz = __square__(video, video.shape[1:3])
#         else:
#             new_sz = tuple(video.shape[1:3])
#
#         # FOURCC codes: http://www.fourcc.org/codecs.php : http://mp4ra.org/#/codecs
#         frame_rate = min(frame_rate, max(1, int(video.shape[0] / 5)))
#         if video.shape[0] > 3:
#             __save_video__(join(dir_save, name + '.mp4'), video, frame_rate, quality=quality, new_sz=new_sz)
#         else:
#             save_jpg = True
#
#     if save_jpg:
#         for z in range(n_slices):
#             imwrite(join(dir_save, name + '-z' + str(z) + '.jpg'), video[z, :, :, :])
#
#     return
