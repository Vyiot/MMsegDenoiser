"""Inference script: denoise pseudo-labels for an entire directory.

Given a directory of satellite images and pseudo-labels, produce
refined (denoised) pseudo-labels using a trained model.

Usage:
    python tools/inference.py \
        configs/denoiser/segformer_b2_512x512_40k_denoise.py \
        work_dirs/segformer_b2_512x512_40k_denoise/latest.pth \
        --img-dir data/test/images \
        --pseudo-dir data/test/pseudo_labels \
        --out-dir data/test/refined_labels \
        --num-classes 7
"""

import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
import mmseg_denoiser  # noqa: F401

from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference: denoise pseudo-labels')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img-dir', required=True, help='Satellite image directory')
    parser.add_argument('--pseudo-dir', required=True, help='Pseudo-label directory')
    parser.add_argument('--out-dir', required=True, help='Output directory for refined labels')
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--pseudo-suffix', default='.png')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def prepare_input(img_path, pseudo_path, num_classes, img_norm_cfg, device):
    """Prepare a single sample for inference.

    Returns:
        Tensor: Combined input (1, 3+num_classes, H, W).
        tuple: Original (H, W) before any processing.
    """
    # Load satellite image
    img = mmcv.imread(img_path)  # (H, W, 3) BGR
    ori_shape = img.shape[:2]

    # Normalize image
    img = img.astype(np.float32)
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # (3, H, W)

    # Load pseudo-label and convert to one-hot
    pseudo = mmcv.imread(pseudo_path, flag='unchanged')
    if pseudo.ndim == 3:
        pseudo = pseudo[:, :, 0]
    pseudo = pseudo.astype(np.int64)

    h, w = pseudo.shape
    onehot = np.zeros((num_classes, h, w), dtype=np.float32)
    valid = pseudo < num_classes
    onehot[pseudo[valid], np.where(valid)[0], np.where(valid)[1]] = 1.0

    # Concatenate: (3 + num_classes, H, W)
    combined = np.concatenate([img, onehot], axis=0)
    tensor = torch.from_numpy(combined).unsqueeze(0).to(device)

    return tensor, ori_shape


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # Build and load model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(args.device)
    model.eval()

    # Get normalization config from dataset config
    img_norm_cfg = cfg.get('img_norm_cfg', dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True))

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect image files
    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.endswith(args.img_suffix)])

    print(f'Found {len(img_files)} images to process.')

    for img_file in tqdm(img_files, desc='Denoising'):
        img_path = osp.join(args.img_dir, img_file)
        pseudo_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        pseudo_path = osp.join(args.pseudo_dir, pseudo_file)

        if not osp.exists(pseudo_path):
            print(f'Warning: pseudo-label not found for {img_file}, skipping.')
            continue

        with torch.no_grad():
            tensor, ori_shape = prepare_input(
                img_path, pseudo_path, args.num_classes,
                img_norm_cfg, args.device)

            # Forward pass
            feats = model.extract_feat(tensor)
            logits = model.decode_head.forward_test(feats, None, None)
            logits = F.interpolate(
                logits, size=ori_shape, mode='bilinear', align_corners=False)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save refined label
        out_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        out_path = osp.join(args.out_dir, out_file)
        Image.fromarray(pred).save(out_path)

    print(f'Done. Refined labels saved to {args.out_dir}')


if __name__ == '__main__':
    main()
