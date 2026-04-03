"""Testing / evaluation script for pseudo-label denoiser.

Usage:
    # Evaluate on validation set
    python tools/test.py \
        configs/denoiser/segformer_b2_512x512_40k_denoise.py \
        work_dirs/segformer_b2_512x512_40k_denoise/latest.pth \
        --eval mIoU

    # Save predictions
    python tools/test.py \
        configs/denoiser/segformer_b2_512x512_40k_denoise.py \
        work_dirs/segformer_b2_512x512_40k_denoise/latest.pth \
        --show-dir work_dirs/predictions/
"""

import os
import warnings
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
warnings.filterwarnings("ignore")

import argparse
import os.path as osp
import sys

import cv2
import mmcv
import torch

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config, DictAction
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
import mmseg_denoiser  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test (evaluate) a pseudo-label denoiser')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+', default=['mIoU'],
                        help='Evaluation metrics (mIoU, mDice, mFscore)')
    parser.add_argument('--show-dir', help='Directory to save prediction visualizations')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id to use')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config options')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # Build model and load checkpoint
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES
                                                    if hasattr(dataset, 'CLASSES') else None)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', dataset.PALETTE
                                                    if hasattr(dataset, 'PALETTE') else None)

    model = MMDataParallel(model, device_ids=[args.gpu_id])

    # Run inference
    results = single_gpu_test(
        model, data_loader,
        show=False,
        out_dir=args.show_dir,
        efficient_test=False)

    # Evaluate
    if args.eval:
        eval_kwargs = dict(metric=args.eval)
        metric = dataset.evaluate(results, **eval_kwargs)
        print('\n' + '=' * 60)
        print('Evaluation Results:')
        print('=' * 60)
        metric_dict = dict(metric)
        for key, val in sorted(metric_dict.items()):
            print(f'  {key}: {val:.4f}')


if __name__ == '__main__':
    main()
