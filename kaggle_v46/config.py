"""Single source of truth for schedules, datasets and evaluation settings."""
import copy
import json
from pathlib import Path

REVISION = "204698b45b3712590f06245fbfba32d3be539812"
CO3D_SPLITS = {
    "train": ["ball/123_14363_28981", "ball/375_42693_85518", "bench/415_57121_110109",
              "bowl/69_5465_12831", "bowl/70_5792_13401", "broccoli/372_41112_81867",
              "broccoli/412_56288_108844", "cake/403_53094_103680"],
    "val": ["bench/415_57112_110099"],
    "test": ["apple/110_13051_23361", "apple/189_20393_38136", "cake/374_42274_84517"],
}
TUM_SPLITS = {
    "train": ["rgbd_dataset_freiburg1_room", "rgbd_dataset_freiburg2_desk",
              "rgbd_dataset_freiburg2_xyz", "rgbd_dataset_freiburg3_long_office_household"],
    "val": ["rgbd_dataset_freiburg1_desk"],
    "test": ["rgbd_dataset_freiburg3_structure_texture_far"],
}


def default_config(dataset):
    assert dataset in ("co3d", "tum")
    return {
        "version": "v46_relative_geometry_calibration", "dataset": dataset, "run_name": "main_seed42",
        "study": "main", "variant": "full", "mode": "train", "seed": 42,
        "eval_seed": 123, "data_root": "", "output_root": "/kaggle/working/vjepa_geo_v46",
        "asset_root": "/kaggle/temp/vjepa_assets", "resume_checkpoint": "",
        "evaluate_checkpoint": "", "allow_download": True,
        "splits": copy.deepcopy(CO3D_SPLITS if dataset == "co3d" else TUM_SPLITS),
        "image_size": 512, "window_views": 10, "eval_views": 5,
        "frame_stride": 4 if dataset == "co3d" else 10, "window_step": 3 if dataset == "co3d" else 5,
        "epochs": [5, 10, 5], "stage_k": [[3, 5, 8], [3, 5], [3, 5, 8]],
        "stage_k_probabilities": [[.45, .35, .20], [.65, .35], [.45, .35, .20]],
        "view_dropout": .15, "micro_batch_per_gpu": 1, "global_effective_batch": 4,
        "workers_per_gpu": 1, "lr_heads": .00015, "lr_backbone": .000002,
        "weight_decay": .0001, "grad_clip": 1., "trainable_backbone_blocks": 1,
        "amp": True, "amp_dtype": "float16", "eval_amp": False,
        "max_hours": 10.5, "require_two_gpus": True, "backend": "nccl",
        "min_depth": .03 if dataset == "co3d" else .1,
        "max_depth": 50. if dataset == "co3d" else 6.,
        "depth_bins": 24, "depth_chunk": 4, "cost_temperature": .15,
        "visibility_abs": .02, "visibility_rel": .03,
        "edge_size": 384, "checkpoint_activations": True,
        "glvp_weight": .025, "glvp_conditioning": True, "glvp_loss": "cosine_contrastive",
        "glvp_target_transform": "train_standardized", "glvp_contrastive_weight": .25,
        "glvp_temperature": .1, "glvp_std_floor": .005,
        "use_epipolar": True, "use_scene_tokens": True, "use_refinement": True,
        "use_render": True, "use_descriptors": True, "use_mask": True,
        "use_confidence_loss": True, "use_confidence_weighting": True,
        "use_pointmap": True, "use_sim3": True, "use_camera_loss": True,
        "use_pairwise": True, "use_reprojection": True, "use_scale_loss": True,
        "render_size": 96, "gaussian_stride": 8, "max_gaussians": 4096,
        "confidence_thresholds": [.005, .01, .02, .03, .05, .1, .2, .4, .6, .8], "mask_threshold": .5,
        "fscore_thresholds": [.05, .10, .20], "cloud_stride": 4,
        "max_metric_points": 50000, "max_fit_points": 20000,
        "reference_support": "object" if dataset == "co3d" else "valid_depth",
        "isolate_intrinsics_gradients": True, "intrinsics_loss_weight": .5,
        "depth_absolute_weight": .25, "depth_relative_weight": 1., "depth_log_weight": .5,
        "confidence_relative_tolerance": .1, "confidence_invalid_weight": .1,
        "pose_normalized_weight": .3, "initial_pose_weight": .1, "refinement_guard_weight": .1,
        "camera_grid": 16, "camera_dim": 128, "camera_match_weight": .1,
        "selection_pose_weight": .5, "selection_rotation_weight": .25,
        "selection_rule": "validation_depth_pose_composite", "selection_alignment": "none",
        "upstream_revision": REVISION,
        "encoder_checkpoint": "vjepa2_1_vitb_dist_vitG_384.pt",
        "encoder_url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "encoder_sha256": "", "tiny_backbone": False, "fixture_manifest": "",
        "export_points": True, "ten_view_trajectory_diagnostic": True,
        "companion_run": "", "max_train_steps": 0, "max_eval_windows": 0,
    }


def apply_study(config):
    c = copy.deepcopy(config)
    if c["study"] == "glvp":
        c["epochs"] = [3, 6, 3]
        c["stage_k"] = [[3], [3], [3]]
        c["stage_k_probabilities"] = [[1.], [1.], [1.]]
        c["eval_views"] = 3
        c["view_dropout"] = 0.
        choices = {"A0_no_glvp": (0., True), "A1_glvp_0025": (.025, True),
                   "A2_no_conditioning": (.025, False), "A3_glvp_0010": (.01, True),
                   "A4_glvp_0050": (.05, True)}
        if c["variant"] not in choices:
            raise ValueError(f"G-LVP variant must be one of {list(choices)}")
        c["glvp_weight"], c["glvp_conditioning"] = choices[c["variant"]]
    elif c["study"] == "components":
        c["epochs"] = [2, 4, 0]
        changes = {
            "full": {}, "frozen_backbone": {"trainable_backbone_blocks": 0},
            "random_frozen_features": {"backbone_initialization": "random", "trainable_backbone_blocks": 0},
            "no_sim3": {"use_sim3": False}, "no_pointmap": {"use_pointmap": False},
            "no_camera": {"use_camera_loss": False}, "no_pairwise": {"use_pairwise": False},
            "no_reprojection": {"use_reprojection": False}, "no_scale": {"use_scale_loss": False},
            "no_render": {"use_render": False}, "no_epipolar": {"use_epipolar": False},
            "no_confidence": {"use_confidence_loss": False, "use_confidence_weighting": False},
            "no_scene_tokens": {"use_scene_tokens": False},
            "no_refinement": {"use_refinement": False}, "no_descriptors": {"use_descriptors": False},
            "single_view": {"stage_k": [[1], [1], [1]], "stage_k_probabilities": [[1.], [1.], [1.]], "eval_views": 1, "view_dropout": 0., "glvp_weight": 0.},
        }
        if c["variant"] not in changes:
            raise ValueError(f"Component variant must be one of {list(changes)}")
        c.update(changes[c["variant"]])
    elif c["study"] == "paired":
        if c["variant"] not in ["full", "no_glvp"]:raise ValueError("paired variants: full, no_glvp")
        if c["variant"] == "no_glvp":c["glvp_weight"]=0.
    elif c["study"] != "main":
        raise ValueError("study must be main, paired, glvp, or components")
    validate(c)
    return c


def validate(c, world_size=None):
    if c['confidence_relative_tolerance']<=0 or not 0<=c['confidence_invalid_weight']<=1:
        raise ValueError('Invalid relative-depth confidence settings')
    if c['glvp_temperature']<=0 or c['glvp_std_floor']<=0 or c['glvp_contrastive_weight']<0:
        raise ValueError('Invalid contrastive target settings')
    if c['glvp_target_transform'] not in ['none','train_standardized']:
        raise ValueError('Unknown teacher target transform')
    if not 1 <= c['camera_grid'] <= c['image_size']//16:
        raise ValueError('Camera grid must fit the encoder patch grid')
    if c['camera_dim'] < 16 or c['camera_dim'] % 4:
        raise ValueError('Camera dimension must be a multiple of four and at least 16')
    if min(c['camera_match_weight'],c['selection_pose_weight'],c['selection_rotation_weight']) < 0:
        raise ValueError('Camera loss and selection weights must be nonnegative')
    if c['selection_rule'] != 'validation_depth_pose_composite':
        raise ValueError('Unsupported checkpoint selection rule')
    if len(c["epochs"]) != 3 or any(int(x) != x or x < 0 for x in c["epochs"]):
        raise ValueError("epochs must be three nonnegative integers")
    if c["mode"] == "train" and sum(c["epochs"]) == 0:
        raise ValueError("Training requires a nonzero explicit schedule")
    if c["micro_batch_per_gpu"] != 1:
        raise ValueError("This memory-bounded configuration supports one window per GPU")
    if c["image_size"] % 16 or c["eval_views"] > c["window_views"]:
        raise ValueError("Invalid image size or evaluation K")
    for choices, probabilities in zip(c["stage_k"], c["stage_k_probabilities"]):
        if len(choices) != len(probabilities) or abs(sum(probabilities)-1.) > 1e-6:
            raise ValueError("Invalid per-stage view probabilities")
        if min(choices) < 1 or max(choices) > c["window_views"]:
            raise ValueError("Training K outside stored window size")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(c["splits"][a]) & set(c["splits"][b]):
            raise ValueError(f"Sequence leakage between {a} and {b}")
    if world_size:
        divisor = world_size * c["micro_batch_per_gpu"]
        if c["global_effective_batch"] % divisor:
            raise ValueError("Global effective batch must be divisible by GPU count × micro batch")


def read_config(path):
    c = json.loads(Path(path).read_text())
    return apply_study(c)
