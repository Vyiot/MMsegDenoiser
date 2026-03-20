# MMSeg Pseudo-Label Denoiser

A pseudo-label denoising framework built on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The model learns to refine noisy pseudo-labels by jointly encoding satellite imagery and one-hot pseudo-label maps through an adapted segmentation backbone.

## Method

The denoiser takes as input the channel-wise concatenation of an RGB satellite image and a one-hot encoded pseudo-label map, producing a shape of `(B, 3 + C, H, W)` where `C` is the number of classes. The backbone's stem convolution is automatically adapted to accept the expanded input channels while preserving ImageNet-pretrained weights for the RGB portion.

Two segmentor architectures are provided:

- **DenoiserSegmentor** &mdash; single-encoder pipeline that passes the concatenated input through one backbone.
- **DualEncoderSegmentor** &mdash; dual-stream architecture with separate backbones for the image and pseudo-label branches, merged via a `FusionNeck`.

Each architecture supports multiple backbones (SegFormer-B2, DeepLabV3+-R101, HRNet-W48, UNet-R50) with optional skip-connection decoding via `UNetDecodeHead`.

## Project Structure

```
├── configs/
│   ├── _base_/
│   │   ├── datasets/pseudo_label_denoise.py    # Dataset & pipeline config
│   │   ├── models/                             # Base model configs (14 variants)
│   │   ├── schedules/schedule_40k.py           # AdamW + poly LR, 40k iters
│   │   └── default_runtime.py                  # Logging, NCCL backend
│   └── denoiser/                               # 16 ready-to-train configs
├── mmseg_denoiser/
│   ├── datasets/
│   │   ├── pseudo_label_dataset.py             # PseudoLabelDenoiseDataset
│   │   └── pipelines.py                        # LoadPseudoLabel, FormatDenoiseBundle
│   ├── models/
│   │   ├── denoiser_segmentor.py               # Single-encoder segmentor
│   │   ├── dual_encoder_segmentor.py           # Dual-encoder segmentor
│   │   ├── denoiser_head.py                    # Denoiser decode head
│   │   ├── unet_decode_head.py                 # UNet-style decode head with skip connections
│   │   └── fusion_neck.py                      # Feature fusion neck for dual-encoder
│   └── losses/
│       ├── symmetric_ce_loss.py                # Symmetric cross-entropy loss
│       └── noise_robust_dice_loss.py           # Noise-robust Dice loss
├── tools/
│   ├── train.py                                # Training entry point
│   ├── test.py                                 # Evaluation entry point
│   ├── inference.py                            # Batch inference on new data
│   └── dist_train.sh                           # Distributed training launcher
└── setup.py
```

## Installation

```bash
# Create environment
conda create -n mmseg-denoiser python=3.8 -y
conda activate mmseg-denoiser

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV and MMSegmentation
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install mmsegmentation==0.30.0

# Install this package
pip install -e .
```

## Data Preparation

Organize your dataset under `data/` as follows:

```
data/my_dataset/
├── images/
│   ├── train/    # Satellite RGB images (.tif)
│   └── val/
├── pseudo_labels/
│   ├── train/    # Noisy pseudo-labels as class-index PNGs
│   └── val/
└── clean_labels/
    ├── train/    # Ground-truth labels as class-index PNGs
    └── val/
```

Then update `configs/_base_/datasets/pseudo_label_denoise.py`:

- `data_root`: path to your dataset root
- `num_classes`: number of segmentation classes
- `img_suffix` / `seg_map_suffix` / `pseudo_label_suffix`: file extensions

## Training

```bash
# Single GPU
python tools/train.py configs/denoiser/segformer_b2_512x512_40k_denoise.py

# Multi-GPU (e.g., 4 GPUs)
./tools/dist_train.sh configs/denoiser/segformer_b2_512x512_40k_denoise.py 4

# Resume from checkpoint
python tools/train.py configs/denoiser/segformer_b2_512x512_40k_denoise.py \
    --resume-from work_dirs/segformer_b2_512x512_40k_denoise/latest.pth

# Override config options
python tools/train.py configs/denoiser/segformer_b2_512x512_40k_denoise.py \
    --cfg-options data.samples_per_gpu=2 optimizer.lr=3e-5
```

## Evaluation

```bash
python tools/test.py \
    configs/denoiser/segformer_b2_512x512_40k_denoise.py \
    work_dirs/segformer_b2_512x512_40k_denoise/latest.pth \
    --eval mIoU
```

## Inference

Denoise pseudo-labels for a directory of images:

```bash
python tools/inference.py \
    configs/denoiser/segformer_b2_512x512_40k_denoise.py \
    work_dirs/segformer_b2_512x512_40k_denoise/latest.pth \
    --img-dir data/test/images \
    --pseudo-dir data/test/pseudo_labels \
    --out-dir data/test/refined_labels \
    --num-classes 7
```

## Available Configurations

| Config | Backbone | Architecture | Skip Conn. |
|--------|----------|-------------|------------|
| `segformer_b2_512x512_40k_denoise` | SegFormer-B2 | Single | No |
| `segformer_b2_512x512_40k_denoise_skipconn` | SegFormer-B2 | Single | Yes |
| `deeplabv3plus_r101_512x512_40k_denoise` | ResNet-101 | Single | No |
| `deeplabv3plus_r101_512x512_40k_denoise_skipconn` | ResNet-101 | Single | Yes |
| `hrnet_w48_512x512_40k_denoise` | HRNet-W48 | Single | No |
| `hrnet_w48_512x512_40k_denoise_skipconn` | HRNet-W48 | Single | Yes |
| `unet_r50_512x512_40k_denoise` | ResNet-50 | Single | No |
| `dual_segformer_b2_512x512_40k_denoise` | SegFormer-B2 | Dual | No |
| `dual_segformer_b2_512x512_40k_denoise_skipconn` | SegFormer-B2 | Dual | Yes |
| `dual_deeplabv3plus_r101_512x512_40k_denoise` | ResNet-101 | Dual | No |
| `dual_deeplabv3plus_r101_512x512_40k_denoise_skipconn` | ResNet-101 | Dual | Yes |
| `dual_hrnet_w48_512x512_40k_denoise` | HRNet-W48 | Dual | No |
| `dual_hrnet_w48_512x512_40k_denoise_skipconn` | HRNet-W48 | Dual | Yes |
| `dual_unet_r50_512x512_40k_denoise` | ResNet-50 | Dual | No |

## License

This project is for research purposes.
