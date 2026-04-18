"""Custom data loading and formatting pipelines for pseudo-label denoising.

These pipelines extend mmseg's standard pipelines to handle the triplet
(satellite image, noisy pseudo-label, clean label) required by the denoiser.
"""

import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPseudoLabel:
    """Load pseudo-label from file and encode as one-hot.

    The pseudo-label is loaded as a single-channel class index map and
    converted to a one-hot representation of shape (num_classes, H, W).
    This one-hot encoding is later concatenated with the satellite image
    along the channel dimension.

    Args:
        to_onehot (bool): Whether to convert to one-hot. Default: True.
            If False, the pseudo-label is kept as a class index map (H, W).
        ignore_index (int): Index to ignore during one-hot encoding.
            Default: 255.
    """

    def __init__(self, to_onehot: bool = False, ignore_index: int = 255):
        self.to_onehot = to_onehot
        self.ignore_index = ignore_index

    def __call__(self, results: dict) -> dict:
        pseudo_label_path = results['pseudo_label_path']
        pseudo_label = mmcv.imread(pseudo_label_path, flag='unchanged')

        if pseudo_label is None:
            raise FileNotFoundError(
                f'Failed to load pseudo-label: {pseudo_label_path}')

        # Ensure 2D (H, W) class index map
        if pseudo_label.ndim == 3:
            pseudo_label = pseudo_label[:, :, 0]

        pseudo_label = pseudo_label.astype(np.int64)
        results['pseudo_label'] = pseudo_label

        if self.to_onehot:
            num_classes = results['num_classes']
            h, w = pseudo_label.shape
            onehot = np.zeros((num_classes, h, w), dtype=np.float32)
            valid_mask = pseudo_label != self.ignore_index
            valid_labels = pseudo_label[valid_mask]
            rows, cols = np.where(valid_mask)
            onehot[valid_labels, rows, cols] = 1.0
            results['pseudo_label_onehot'] = onehot

        results['seg_fields'].append('pseudo_label')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'to_onehot={self.to_onehot}, '
                f'ignore_index={self.ignore_index})')



@PIPELINES.register_module()
class PseudoLabelToOneHot:
    """Convert pseudo-label class index map to one-hot encoding.

    This should be placed in the pipeline after all spatial transformations
    (Resize, RandomCrop, etc.) to ensure the one-hot representation
    matches the transformed image dimensions.

    Args:
        ignore_index (int): Index to ignore during one-hot encoding.
            Default: 255.
    """

    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index

    def __call__(self, results: dict) -> dict:
        if 'pseudo_label' not in results:
            return results

        pseudo_label = results['pseudo_label']
        num_classes = results['num_classes']
        h, w = pseudo_label.shape

        onehot = np.zeros((num_classes, h, w), dtype=np.float32)
        valid_mask = pseudo_label != self.ignore_index
        valid_labels = pseudo_label[valid_mask]
        rows, cols = np.where(valid_mask)

        # Handle indexing for one-hot
        onehot[valid_labels, rows, cols] = 1.0

        results['pseudo_label_onehot'] = onehot
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(ignore_index={self.ignore_index})'


@PIPELINES.register_module()
class FormatDenoiseBundle:
    """Format the data bundle for pseudo-label denoising.

    Concatenates the satellite image (C, H, W) with the one-hot
    pseudo-label (num_classes, H, W) to produce a combined input tensor
    of shape (C + num_classes, H, W).

    Also formats gt_semantic_seg for loss computation.

    Args:
        img_to_float (bool): Whether to convert image to float32. Default: True.
    """

    def __init__(self, img_to_float: bool = True):
        self.img_to_float = img_to_float

    def __call__(self, results: dict) -> dict:
        # Image: (H, W, C) -> (C, H, W)
        img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose(2, 0, 1)  # (C, H, W)

        if self.img_to_float:
            img = img.astype(np.float32)

        # Pseudo-label one-hot: already (num_classes, H, W)
        pseudo_onehot = results.get('pseudo_label_onehot')
        if pseudo_onehot is not None:
            # Concatenate along channel dim: (C + num_classes, H, W)
            combined = np.concatenate([img, pseudo_onehot], axis=0)
        else:
            combined = img

        results['img'] = torch.from_numpy(combined).contiguous()

        # Format ground-truth segmentation label
        if 'gt_semantic_seg' in results:
            gt = results['gt_semantic_seg']
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt = gt.astype(np.int64)
            results['gt_semantic_seg'] = torch.from_numpy(
                gt[None, ...]).long().contiguous()  # (1, H, W)

        # Keep pseudo_label as class indices for evaluation
        if 'pseudo_label' in results:
            results['pseudo_label_indices'] = torch.from_numpy(
                results['pseudo_label']).long().contiguous()

        # Format timestep as tensor (for diffusion mode)
        if 'timestep' in results:
            results['timesteps'] = torch.tensor(
                results['timestep'], dtype=torch.long)

        # Collect meta information
        img_meta = {}
        for key in ['img_info', 'ori_shape', 'img_shape', 'pad_shape',
                     'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                     'num_classes']:
            if key in results:
                img_meta[key] = results[key]
        results['img_metas'] = img_meta

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(img_to_float={self.img_to_float})'
