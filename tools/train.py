"""Training script for pseudo-label denoiser.

Usage:
    # Single GPU
    python tools/train.py configs/denoiser/segformer_b2_512x512_40k_denoise.py

    # Multi-GPU (e.g., 4 GPUs)
    ./tools/dist_train.sh configs/denoiser/segformer_b2_512x512_40k_denoise.py 4

    # Resume from checkpoint
    python tools/train.py configs/denoiser/segformer_b2_512x512_40k_denoise.py \
        --resume-from work_dirs/segformer_b2_512x512_40k_denoise/latest.pth
"""

import argparse
import copy
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

# Register custom modules
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
import mmseg_denoiser  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a pseudo-label denoiser')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume-from', help='Checkpoint to resume from')
    parser.add_argument('--load-from', help='Checkpoint to load weights from (no optimizer state)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Whether not to evaluate during training')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (only for non-distributed)')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                        help='IDs of GPUs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Whether to set deterministic options for CUDNN')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config options. Format: key=value')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # Init distributed training
    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)
        distributed = True

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # Init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seed
    seed = args.seed
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    # Build model
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(model)

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # Add metadata
    model.CLASSES = datasets[0].CLASSES if hasattr(datasets[0], 'CLASSES') else None
    model.PALETTE = datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None

    # Train
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=dict(
            env_info=env_info,
            seed=seed,
            config=cfg.pretty_text))


if __name__ == '__main__':
    main()
