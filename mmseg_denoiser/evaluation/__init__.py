"""Custom evaluation hook for diffusion denoiser with dual validation.

Runs both single-step and iterative evaluation during training,
logging both metrics in a comparison table (similar to SegRefiner).
"""

import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class DiffusionEvalHook(Hook):
    """Dual evaluation hook: single-step + iterative metrics.

    Args:
        dataloader: Validation dataloader.
        interval (int): Evaluation interval (iterations).
        label_dir (str): Path to ground-truth labels.
        pseudo_dir (str): Path to pseudo-labels (for pseudo IoU reference).
    """

    def __init__(self, dataloader, interval=4000,
                 label_dir='', pseudo_dir=''):
        self.dataloader = dataloader
        self.interval = interval
        self.label_dir = label_dir
        self.pseudo_dir = pseudo_dir
        self.best_miou = -1.0
        self.best_iter = -1

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._do_evaluate(runner)

    def _load_mask(self, path):
        mask = mmcv.imread(path, flag='unchanged')
        if mask is None:
            raise FileNotFoundError(f'Cannot load mask: {path}')
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8)

    def _compute_iou(self, pred, gt):
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        return inter, union

    def _do_evaluate(self, runner):
        runner.logger.info(
            f'\n--- Dual evaluation at iter {runner.iter + 1} ---')
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        model.eval()

        # Accumulators: single/iter/pseudo × building/bg
        s_ri, s_ru, s_ri_bg, s_ru_bg = 0, 0, 0, 0
        it_ri, it_ru, it_ri_bg, it_ru_bg = 0, 0, 0, 0
        p_ri, p_ru, p_ri_bg, p_ru_bg = 0, 0, 0, 0
        total = 0

        prog_bar = mmcv.ProgressBar(len(self.dataloader.dataset))

        for data in self.dataloader:
            with torch.no_grad():
                # Unwrap DataContainer from mmseg dataloader
                img = data['img'].data[0]
                if not img.is_cuda:
                    img = img.cuda()
                img_metas = data['img_metas'].data[0]

                results = model.diffusion_test(img, img_metas, rescale=True)

            # results = [(pred, 'single'), (pred, 'iterative'), ...] per sample
            singles = [m for m, mode in results if mode == 'single']
            iters = [m for m, mode in results if mode == 'iterative']

            for sample_idx, (s_mask, it_mask) in enumerate(zip(singles, iters)):
                # Get filename from img_meta
                meta = img_metas[sample_idx]
                filename = osp.basename(meta['img_info']['filename'])
                basename = osp.splitext(filename)[0]

                gt_path = osp.join(self.label_dir, basename + '.tif')
                pseudo_path = osp.join(self.pseudo_dir, basename + '.tif')

                if not osp.exists(gt_path):
                    gt_path = osp.join(self.label_dir, filename)
                if not osp.exists(pseudo_path):
                    pseudo_path = osp.join(self.pseudo_dir, filename)

                gt = self._load_mask(gt_path)
                pseudo = self._load_mask(pseudo_path)

                # Ensure shapes match
                if s_mask.shape != gt.shape:
                    s_mask = cv2.resize(s_mask.astype(np.uint8),
                                        (gt.shape[1], gt.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                if it_mask.shape != gt.shape:
                    it_mask = cv2.resize(it_mask.astype(np.uint8),
                                         (gt.shape[1], gt.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                if pseudo.shape != gt.shape:
                    pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

                s_pred = (s_mask > 0).astype(np.uint8)
                it_pred = (it_mask > 0).astype(np.uint8)

                # Building IoU (class 1)
                i, u = self._compute_iou(s_pred, gt); s_ri += i; s_ru += u
                i, u = self._compute_iou(it_pred, gt); it_ri += i; it_ru += u
                i, u = self._compute_iou(pseudo, gt); p_ri += i; p_ru += u

                # Background IoU (class 0)
                i, u = self._compute_iou(1 - s_pred, 1 - gt); s_ri_bg += i; s_ru_bg += u
                i, u = self._compute_iou(1 - it_pred, 1 - gt); it_ri_bg += i; it_ru_bg += u
                i, u = self._compute_iou(1 - pseudo, 1 - gt); p_ri_bg += i; p_ru_bg += u

                total += 1
                prog_bar.update()

        # Compute metrics
        s_iou_b = s_ri / max(s_ru, 1)
        s_iou_bg = s_ri_bg / max(s_ru_bg, 1)
        s_miou = (s_iou_b + s_iou_bg) / 2

        it_iou_b = it_ri / max(it_ru, 1)
        it_iou_bg = it_ri_bg / max(it_ru_bg, 1)
        it_miou = (it_iou_b + it_iou_bg) / 2

        p_iou_b = p_ri / max(p_ru, 1)
        p_iou_bg = p_ri_bg / max(p_ru_bg, 1)
        p_miou = (p_iou_b + p_iou_bg) / 2

        # Log
        runner.log_buffer.output['val/mIoU_single'] = s_miou
        runner.log_buffer.output['val/mIoU_iter'] = it_miou
        runner.log_buffer.output['val/mIoU_pseudo'] = p_miou

        header = f'{"Class":<12} {"Single IoU":>12} {"Iter IoU":>12} {"Pseudo IoU":>12}'
        sep = '-' * len(header)
        rows = [
            sep, header, sep,
            f'{"background":<12} {s_iou_bg*100:>11.2f}% {it_iou_bg*100:>11.2f}% {p_iou_bg*100:>11.2f}%',
            f'{"building":<12} {s_iou_b*100:>11.2f}% {it_iou_b*100:>11.2f}% {p_iou_b*100:>11.2f}%',
            sep,
            f'{"mIoU":<12} {s_miou*100:>11.2f}% {it_miou*100:>11.2f}% {p_miou*100:>11.2f}%',
            sep,
        ]
        runner.logger.info('\n' + '\n'.join(rows))
        runner.log_buffer.ready = True

        # Save best checkpoint
        if s_miou > self.best_miou:
            self.best_miou = s_miou
            self.best_iter = runner.iter + 1
            runner.logger.info(
                f'New best mIoU_single: {s_miou*100:.2f}% at iter {self.best_iter}. '
                f'Saving checkpoint...')
            runner.save_checkpoint(
                runner.work_dir, filename_tmpl='best_mIoU_single.pth',
                create_symlink=False)

        model.train()
