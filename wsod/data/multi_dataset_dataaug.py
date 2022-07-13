import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image


from detectron2.data import transforms as T
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

def build_multi_augmentation(is_train, scale, size):
    '''
    Create a list of default :class:`Augmentation` from a python script
    which used to create a dataset

    Returns:
        List[Augmentation]
    '''
    if is_train:
        scale = scale
        size = size
    else:
        scale = (1, 1)
        size = size
    
    augmentation = [EfficientDetResizeCrop(size, scale)]

    if is_train:
        augmentation.append(T.RandomFlip())

    return augmentation
        