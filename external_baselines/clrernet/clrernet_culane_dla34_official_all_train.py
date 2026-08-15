"""CLRerNet v0.3 DLA34 on the untouched official CULane train/val lists.

This is intentionally an overlay on the upstream v0.3.0 configuration.  The
upstream recipe removes low-frame-difference training samples with
``train_diffs.npz``.  That behavior is explicitly disabled here: every one of
the 88,880 rows in ``train_gt.txt`` participates in training.

The relative base path assumes the two repositories are siblings named
``DynLaneSeq`` and ``CLRerNet_official`` under the same workspace.
"""

_base_ = [
    "../../../CLRerNet_official/configs/clrernet/culane/"
    "clrernet_culane_dla34.py"
]

data_root = "/workspace/CULane"
work_dir = "/workspace/CLRerNet_runs/culane_dla34_v030_all_official_train"

# Keep the upstream published deployment threshold fixed.  It is not selected
# on our validation set.
model = dict(test_cfg=dict(conf_threshold=0.41))

total_epochs = 15
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=total_epochs,
    # Endpoint-only validation: no intermediate validation checkpoint choice.
    val_interval=total_epochs,
)

train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_list=data_root + "/list/train_gt.txt",
        # Critical protocol difference from the upstream published recipe.
        diff_file=None,
        diff_thr=0,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_list=data_root + "/list/val.txt",
    ),
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_list=data_root + "/list/test.txt",
    ),
)

val_evaluator = dict(
    data_root=data_root,
    data_list=data_root + "/list/val.txt",
)
test_evaluator = dict(
    data_root=data_root,
    data_list=data_root + "/list/test.txt",
)

# Save only the fixed endpoint.  ``save_best`` is deliberately disabled.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=total_epochs,
        save_best=None,
        max_keep_ckpts=1,
    )
)

randomness = dict(seed=0, deterministic=True)
load_from = None
resume = False

