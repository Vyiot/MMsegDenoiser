"""Dataset for pseudo-label denoising.

This dataset loads triplets of:
    - Satellite image (RGB, condition)
    - Noisy pseudo-label (from a poorly trained segmentation model)
    - Clean ground-truth label (supervision target)

The satellite image and the one-hot encoded pseudo-label are concatenated
along the channel dimension to form the network input.
"""

import os
import os.path as osp
from typing import Dict, List, Optional, Sequence

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class PseudoLabelDenoiseDataset(CustomDataset):
    """Dataset for pseudo-label denoising task.

    Directory structure::

        data_root/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── pseudo_labels/
        │   ├── train/
        │   └── val/
        └── clean_labels/
            ├── train/
            └── val/

    Args:
        pipeline (list[dict]): Processing pipeline.
        img_dir (str): Path to satellite image directory.
        pseudo_label_dir (str): Path to noisy pseudo-label directory.
        ann_dir (str): Path to clean ground-truth label directory.
        num_classes (int): Number of segmentation classes.
        img_suffix (str): Suffix of satellite images. Default: '.tif'.
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'.
        pseudo_label_suffix (str): Suffix of pseudo-label files. Default: '.png'.
        split (str, optional): Split file path.
        data_root (str, optional): Data root for img_dir, ann_dir, pseudo_label_dir.
        reduce_zero_label (bool): Whether to reduce zero label. Default: False.
        classes (list[str], optional): Class names.
        palette (list[list[int]], optional): Palette for visualization.
        ignore_index (int): Ignore index for loss computation. Default: 255.
    """

    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self,
                 pipeline: List[dict],
                 img_dir: str,
                 pseudo_label_dir: str,
                 ann_dir: str,
                 num_classes: int,
                 img_suffix: str = '.tif',
                 seg_map_suffix: str = '.png',
                 pseudo_label_suffix: str = '.png',
                 split: Optional[str] = None,
                 data_root: Optional[str] = None,
                 reduce_zero_label: bool = False,
                 classes: Optional[Sequence[str]] = None,
                 palette: Optional[List[List[int]]] = None,
                 ignore_index: int = 255,
                 test_mode: bool = False,
                 use_all_t: bool = False,
                 noise_mask_dir: str = 'noise_mask',
                 num_timesteps: int = 6,
                 **kwargs):
        self.num_classes = num_classes
        self.pseudo_label_dir = pseudo_label_dir
        self.pseudo_label_suffix = pseudo_label_suffix
        self.use_all_t = use_all_t
        self.noise_mask_dir = noise_mask_dir
        self.num_timesteps = num_timesteps
        self._noise_sources = ['noise_gen', 'mix']
        self._samples_per_image = num_timesteps * len(self._noise_sources)  # 12

        # Initialize parent (CustomDataset handles img_dir, ann_dir, etc.)
        super().__init__(
            pipeline=[],  # We set pipeline later
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            split=split,
            data_root=data_root,
            test_mode=test_mode,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            palette=palette,
            ignore_index=ignore_index,
            **kwargs)

        # Resolve pseudo_label_dir path
        if self.data_root is not None and not osp.isabs(self.pseudo_label_dir):
            self.pseudo_label_dir = osp.join(self.data_root, self.pseudo_label_dir)

        # Resolve noise_mask_dir path
        if self.data_root is not None and not osp.isabs(self.noise_mask_dir):
            self.noise_mask_dir = osp.join(self.data_root, self.noise_mask_dir)

        # Override pipeline with our custom pipeline
        self.pipeline = Compose(pipeline)

        if self.use_all_t:
            print_log(
                f'use_all_t=True: {len(self.img_infos)} images x '
                f'{self._samples_per_image} = {len(self)} samples',
                logger='root')

    def __len__(self):
        if self.use_all_t:
            return len(self.img_infos) * self._samples_per_image
        return len(self.img_infos)

    def prepare_train_img(self, idx: int) -> Dict:
        """Get training data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data dict.
        """
        if self.use_all_t:
            # Decode expanded index: (image_idx, source, timestep)
            img_idx = idx // self._samples_per_image
            remainder = idx % self._samples_per_image
            source_idx = remainder // self.num_timesteps
            t = remainder % self.num_timesteps
            source = self._noise_sources[source_idx]

            img_info = self.img_infos[img_idx]
            ann_info = self.get_ann_info(img_idx)

            # noise_mask/{source}/{t}/{filename}.tif
            base_name = osp.splitext(img_info['filename'])[0]
            noise_mask_path = osp.join(
                self.noise_mask_dir, source, str(t), base_name + '.tif')

            results = dict(
                img_info=img_info,
                ann_info=ann_info,
                seg_fields=[],
                img_prefix=self.img_dir,
                seg_prefix=self.ann_dir,
                pseudo_label_path=noise_mask_path,
                num_classes=self.num_classes,
                timestep=t,
            )
            self.pre_pipeline(results)
            return self.pipeline(results)
        else:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)

            pseudo_label_filename = img_info['filename'].replace(
                self.img_suffix, self.pseudo_label_suffix)
            pseudo_label_path = osp.join(
                self.pseudo_label_dir, pseudo_label_filename)

            results = dict(
                img_info=img_info,
                ann_info=ann_info,
                seg_fields=[],
                img_prefix=self.img_dir,
                seg_prefix=self.ann_dir,
                pseudo_label_path=pseudo_label_path,
                num_classes=self.num_classes,
            )
            self.pre_pipeline(results)
            return self.pipeline(results)

    def prepare_test_img(self, idx: int) -> Dict:
        """Get testing data after pipeline."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        pseudo_label_filename = img_info['filename'].replace(
            self.img_suffix, self.pseudo_label_suffix)
        pseudo_label_path = osp.join(self.pseudo_label_dir, pseudo_label_filename)

        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            seg_fields=[],
            img_prefix=self.img_dir,
            seg_prefix=self.ann_dir,
            pseudo_label_path=pseudo_label_path,
            num_classes=self.num_classes,
        )
        self.pre_pipeline(results)
        return self.pipeline(results)
