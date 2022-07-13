# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import json
import logging
from detectron2 import data
import numpy as np
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util

from detectron2.config import configurable

from .custom_build_augmentation import build_custom_augmentation
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import Keypoints, PolygonMasks, BitMasks
from fvcore.transforms.transform import TransformList
from .multi_dataset_dataaug import build_multi_augmentation
from .tar_dataset import DiskTarDataset
from IPython import embed
import random

__all__ = ["CustomDatasetMapper"]

class CustomDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, is_train: bool, 
        with_ann_type=False,
        dataset_ann=[],
        use_diff_bs_size=False,
        dataset_augs=[],
        is_debug=False,
        use_tar_dataset=False,
        tarfile_path='',
        tar_index_dir='',
        **kwargs):
        """
        add image labels
        """
        self.with_ann_type = with_ann_type
        self.dataset_size = 640
        self.dataset_ann = dataset_ann
        self.use_diff_bs_size = use_diff_bs_size
        if self.use_diff_bs_size:
            dataset_info = {}
            dataset_info["scale_range"] = [[0.1, 2.0], [0.5, 1.5]]
            dataset_info["train_size"] = [640, 320]
            dataset_info["test_size"] = 640
            dataset_augs = [
                build_multi_augmentation(True, scale, size) for scale, size in zip(dataset_info["scale_range"], dataset_info["train_size"])
            ]
    
        if self.use_diff_bs_size and is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        self.is_debug = is_debug
        self.use_tar_dataset = use_tar_dataset
        if self.use_tar_dataset:
            print('Using tar dataset')
            self.tar_dataset = DiskTarDataset(tarfile_path, tar_index_dir)
        super().__init__(is_train, **kwargs)
 

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            'with_ann_type': cfg.WITH_IMAGE_LABELS,
            'dataset_ann': cfg.DATALOADER.DATASET_ANN,
            'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
            'is_debug': cfg.IS_DEBUG,
            'use_tar_dataset': cfg.DATALOADER.USE_TAR_DATASET,
            'tarfile_path': cfg.DATALOADER.TARFILE_PATH,
            'tar_index_dir': cfg.DATALOADER.TAR_INDEX_DIR,
        })
        if ret['use_diff_bs_size'] and is_train:
            if cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
                dataset_scales = cfg.DATALOADER.DATASET_INPUT_SCALE
                dataset_sizes = cfg.DATALOADER.DATASET_INPUT_SIZE
                ret['dataset_augs'] = [
                    build_custom_augmentation(cfg, True, scale, size) \
                        for scale, size in zip(dataset_scales, dataset_sizes)]
                
            else:
                assert cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge'
                min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
                max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
                ret['dataset_augs'] = [
                    build_custom_augmentation(
                        cfg, True, min_size=mi, max_size=ma) \
                        for mi, ma in zip(min_sizes, max_sizes)]
        else:
            ret['dataset_augs'] = []

        return ret

    def __call__(self, dataset_dict):
        """
        include image labels
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        h, w = dataset_dict['height'], dataset_dict['width']

        if 'file_name' in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format)
        else:
            ori_image, _, _ = self.tar_dataset[dataset_dict["tar_index"]]
            ori_image = utils._apply_exif_orientation(ori_image)
            ori_image = utils.convert_PIL_to_numpy(ori_image, self.image_format)
        utils.check_image_size(dataset_dict, ori_image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        if self.is_debug:
            dataset_dict['dataset_source'] = 0

        not_full_labeled = 'dataset_source' in dataset_dict and \
            self.with_ann_type and \
                self.dataset_ann[dataset_dict['dataset_source']] != 'box'
        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=sem_seg_gt)
        if self.use_diff_bs_size and self.is_train:
            transforms = \
                self.dataset_augs[dataset_dict['dataset_source']](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, 
                proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            all_annos = [
                (utils.transform_instance_annotations(
                    obj, transforms, image_shape, 
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                ),  obj.get("iscrowd", 0))
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            
            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        if self.with_ann_type:
            dataset_dict["pos_category_ids"] = dataset_dict.get(
                'pos_category_ids', [])
            dataset_dict["ann_type"] = \
                self.dataset_ann[dataset_dict['dataset_source']]
        if self.is_debug and (('pos_category_ids' not in dataset_dict) or \
            (dataset_dict['pos_category_ids'] == [])):
            dataset_dict['pos_category_ids'] = [x for x in sorted(set(
                dataset_dict['instances'].gt_classes.tolist()
            ))]
        return dataset_dict

    def reshape_image(self, ori_input, dataset_dict = None, seg = None, bbox=None):
        resize_scale = random.uniform(0.3, 1.5)
        h, w = ori_input.shape[:2]
        new_h = int(h * resize_scale)
        new_w = int(w * resize_scale)
        
        # ori_image_reshape
        if dataset_dict is not None:
            augs = T.AugmentationList([
                T.RandomFlip(prob=0.5),
                T.ResizeTransform(h, w, new_h, new_w),
            ])
            input = T.AugInput(ori_input, sem_seg=seg)
            transform = augs(input)  # type: T.Transform
            image_transformed = input.image  # new image
            seg_transformed = input.sem_seg

            image_shape = image_transformed.shape[:2]
            all_annos = [
                    (utils.transform_instance_annotations(
                        obj, transform, image_shape, 
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    ),  obj.get("iscrowd", 0))
                    for obj in dataset_dict.pop("annotations")
                ]
            # new annotations
            annos = [ann[0] for ann in all_annos if ann[1] == 0]

            bbox_transformed = np.asarray([obj['bbox'] for obj in annos])

            ins_cls = np.array([obj['category_id'] for obj in annos])

            new_image = np.zeros((self.dataset_size, self.dataset_size, 3))
            new_seg = np.zeros((self.dataset_size, self.dataset_size))

            # reshape_image_size < dataset_size
            if new_h <= self.dataset_size and new_w <= self.dataset_size:
                local_h = random.randint(0, self.dataset_size - new_h)
                local_w = random.randint(0, self.dataset_size - new_w)
                new_image[local_h:local_h+new_h, local_w:local_w+new_w, :] += image_transformed
                new_seg[local_h:local_h+new_h, local_w:local_w+new_w] += seg_transformed
                if len(bbox_transformed) > 1:
                    bbox_transformed[:, [0, 2]] += local_w
                    bbox_transformed[:, [1, 3]] += local_h

            # reshape_image_size > dataset_size
            else:
                res_h = self.dataset_size - new_h if (self.dataset_size - new_h > 0) else 0
                res_w = self.dataset_size - new_w if (self.dataset_size - new_w > 0) else 0
                offset_h = random.randint(0, res_h)
                offset_w = random.randint(0, res_w)
                temp_image = image_transformed[offset_h:offset_h + self.dataset_size, offset_w:offset_w + self.dataset_size, :]
                temp_seg = seg_transformed[offset_h:offset_h + self.dataset_size, offset_w:offset_w + self.dataset_size]
                new_image[:temp_image.shape[0], :temp_image.shape[1], :] += temp_image
                new_seg[:temp_seg.shape[0], :temp_seg.shape[1]] += temp_seg

                if len(bbox_transformed) > 1:
                    bbox_transformed[:, [0, 2]] -= offset_w
                    bbox_transformed[:, [1, 3]] -= offset_h
                    bbox_transformed = np.clip(bbox_transformed, 0, self.dataset_size)
                    
                    # some instance may not in image
                    mask = (bbox_transformed[:, 2] - bbox_transformed[:, 0]) * (bbox_transformed[:, 3] - bbox_transformed[:, 1]) > 0
                    bbox_transformed = bbox_transformed[mask]
                    ins_cls = ins_cls[mask]

            return np.transpose(new_image, (2, 0, 1)), bbox_transformed, new_seg, ins_cls
        
        # ImageNet_data reshape
        else:
            augs = T.AugmentationList([
            T.RandomFlip(prob=0.5),
            T.ResizeTransform(h, w, 224, 224),
        ])
            input = T.AugInput(ori_input, sem_seg=seg, boxes=bbox)
            transform = augs(input)  # type: T.Transform
            image_transformed = input.image  # new image
            seg_transformed = input.sem_seg
            bbox_transformed = input.boxes
            new_image = image_transformed[int(bbox_transformed[0][1]):int(bbox_transformed[0][3] + 1), int(bbox_transformed[0][0]):int(bbox_transformed[0][2] + 1),:]
            new_seg = seg_transformed[int(bbox_transformed[0][1]):int(bbox_transformed[0][3] + 1), int(bbox_transformed[0][0]):int(bbox_transformed[0][2] + 1)]
            bbox_transformed[:,[0, 1]] = 0
            h, w = new_image.shape[:2]
            bbox_transformed[:,[2, 3]] = new_image.shape[:2][::-1]
            return np.transpose(new_image, (2, 0, 1)), bbox_transformed, new_seg

# DETR augmentation
def build_transform_gen(cfg, is_train):
    """
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict