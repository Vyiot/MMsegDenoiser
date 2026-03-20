_base_ = [
    '../_base_/models/dual_segformer_b2_skipconn.py',
    '../_base_/datasets/pseudo_label_denoise.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

optimizer = dict(lr=6e-5, weight_decay=0.01)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
