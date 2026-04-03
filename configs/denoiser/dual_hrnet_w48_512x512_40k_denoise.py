_base_ = [
    '../_base_/models/dual_hrnet_w48.py',
    '../_base_/datasets/pseudo_label_denoise.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

optimizer = dict(lr=1e-4, weight_decay=0.0005)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
device = 'cuda'