import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
import sys
#sys.path.insert(0, '/share/home/ytcheng/Code/MIMDet/')
from wsod.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop
from wsod.data.custom_dataset_mapper import CustomDatasetMapper
from detectron2.evaluation import LVISEvaluator
from detectron2.data.samplers import RepeatFactorTrainingSampler
from omegaconf import OmegaConf
from IPython import embed

dataloader = OmegaConf.create()
# sampler_info = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset_dicts=get_detection_dataset_dicts(names="lvis_v1_train"),
#             repeat_thresh=0.001,)

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_train"),
    mapper=L(CustomDatasetMapper)(
        is_train=True,
        with_ann_type=False,
        dataset_ann=['box', 'box'],
        use_diff_bs_size=False,
        dataset_augs=[],
        augmentations=[L(EfficientDetResizeCrop)(
            size=640,
            scale=[0.1, 2.0],
        ),
        L(T.RandomFlip)(horizontal=True),
        ],
        image_format="RGB",
        use_instance_mask=True,
    ),
    sampler = L(RepeatFactorTrainingSampler)(
        repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
            dataset_dicts = L(get_detection_dataset_dicts)(names="lvis_v1_train"),
            repeat_thresh=0.001
        )
    ),
    total_batch_size=16,
    num_workers=32,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[L(EfficientDetResizeCrop)(size=640, scale=[0.1, 2.0]),],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=32,
)


dataloader.evaluator = L(LVISEvaluator)(dataset_name="${..test.dataset.names}",)
