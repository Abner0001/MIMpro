from detectron2.data.detection_utils import filter_empty_instances
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import LVISEvaluator
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '/share/home/ytcheng/Code/MIMDet/')
from wsod.data.custom_dataset_mapper import CustomDatasetMapper
from wsod.data.custom_dataset_dataloader import build_custom_train_loader, get_detection_dataset_dicts_with_source, MultiDatasetSampler
from wsod.data.custom_build_augmentation import build_custom_augmentation
from wsod.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop

from IPython import embed


dataloader = OmegaConf.create()

# def create_cfg(dataset_names):
#     dataset_dicts = get_detection_dataset_dicts_with_source(
#         dataset_names=dataset_names,
#         filter_empty=False,
#         min_keypoints=0,
#         proposal_files=None,
#     )
#     sampler = MultiDatasetSampler(
#         dataset_dicts,
#         dataset_ratio=[1, 4],
#         use_rfs=[True, False],
#         dataset_ann=['box', 'image'],
#         repeat_threshold=0.001,
#     )

#     mapper = CustomDatasetMapper(
#         is_train=True,
#         with_ann_type=True,
#         dataset_ann=['box', 'image'],
#         use_diff_bs_size=True,
#         dataset_augs=[],
#         image_format="RGB",
#         use_instance_mask=False,
#         augmentations=[],
#     )

#     return {
#         "dataset": dataset_dicts,
#         "sampler": sampler,
#         "mapper": mapper
#     }

# config = create_cfg(("lvis_v1_train","imagenet_lvis_v1"))

dataloader.train = L(build_custom_train_loader)(
    dataset=L(get_detection_dataset_dicts_with_source)(
        dataset_names=("lvis_v1_train","imagenet_lvis_v1"),
        filter_empty=False,
        min_keypoints=0,
        proposal_files=None,
    ),
    mapper=L(CustomDatasetMapper)(
        is_train=True,
        with_ann_type=True,
        dataset_ann=['box', 'image'],
        use_diff_bs_size=True,
        dataset_augs=[],
        augmentations=[L(EfficientDetResizeCrop)(
            size=640,
            scale=[0.1, 2.0],
        ),
        L(T.RandomFlip)(horizontal=True),
        ],
        mage_format="RGB",
        use_instance_mask=False,
    ),
    sampler = L(MultiDatasetSampler)(
            dataset_dicts=L(get_detection_dataset_dicts_with_source)(
            dataset_names=("lvis_v1_train","imagenet_lvis_v1"),
            filter_empty=False,
            min_keypoints=0,
            proposal_files=None,
        ),
        dataset_ratio=[1, 4],
        use_rfs=[True, False],
        dataset_ann=['box', 'image'],
        repeat_threshold=0.001,
    ),
    num_datasets=2,
    multi_dataset_grouping=True,
    dataset_bs=[8, 32],
    use_diff_bs_size=True,
    total_batch_size=16,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=8,
)