# Multi-View 3D Reconstruction Using V-JEPA Features

**Anwar Shamim · Jeffrin Sam**

Training and evaluation source code for multi-view reconstruction using
V-JEPA 2.1 features. CO3D and TUM RGB-D use independently trained models.

## Choose a notebook

| Dataset | Final standalone notebook | Readable Python implementation |
| --- | --- | --- |
| CO3D — relative objectives | [train_co3d_2gpu.ipynb](kaggle_v46/train_co3d_2gpu.ipynb) | [kaggle_v46/](kaggle_v46/) |
| TUM RGB-D — global camera | [train_tum_rgbd_2gpu.ipynb](kaggle_v44/train_tum_rgbd_2gpu.ipynb) | [kaggle_v44/](kaggle_v44/) |

Choose the dataset you want to train. The two notebooks are independent;
neither requires the other to finish. The package names are retained for
import and checkpoint compatibility.

## Run on Kaggle

1. Import the chosen notebook into Kaggle.
2. Select **GPU T4 ×2** and enable **Internet**.
3. Keep `MODE = 'train'`, `STUDY = 'main'`, `VARIANT = 'full'` and
   `ALLOW_DOWNLOAD = True` for the primary training configuration.
4. Run the cells in order. Each notebook embeds its complete Python package.
5. Check `run_state.json` for completion and save the outputs before ending
   the session.

**No dataset attachment is required for fresh training.** Missing dataset
files and encoder assets download automatically into `/kaggle/temp`:

| Material | Default location |
| --- | --- |
| CO3D data | `/kaggle/temp/co3d_data` |
| TUM RGB-D data | `/kaggle/temp/datasets/tum_rgbd` |
| Pretrained encoder assets | `/kaggle/temp/vjepa_assets` |
| Checkpoints, tables and figures | Under `/kaggle/working`, in the selected notebook's output directory |

## Primary training settings

| Parameter | Value |
| --- | --- |
| GPUs | 2; one training process per GPU |
| Schedule | 20 epochs: 5 frozen-encoder, 10 partial fine-tuning, 5 alignment/refinement |
| Image size | 512 × 512 |
| Effective batch | 4 windows: 2 GPUs × microbatch 1 × accumulation 2 |
| Training views | Stage-dependent sampling of 3, 5 or 8 views |
| Primary evaluation | 5 views from each 10-frame window |
| Prediction-module learning rate | `1.5e-4` |
| Encoder learning rate | `2e-6` |
| Seed | 42 |
| Session time budget | 10.5 hours |

The notebook evaluates the validation-selected model after a completed
training schedule and exports measurements, figures and point clouds.
The selected model may precede the last training epoch. Configuration and
sequence splits are defined in each package's `config.py` and saved in the
run's `resolved_config.json` and `data_manifest.json`.

## Continue training or run evaluation

If the time budget is reached before all epochs finish, save both `last.pt`
and its sibling `best.pt`. Make those outputs accessible in the next session,
use the same source and scientific settings, set `MODE = 'resume'`, and point
`RESUME_CHECKPOINT` to the saved `last.pt`.

For inference with a compatible trained model, set `MODE = 'evaluate'` and
`EVALUATE_CHECKPOINT` to its `best.pt`. Checkpoints from different
configurations are not interchangeable. Fresh training downloads its data;
continuation and evaluation require access to the relevant trained model.

## Source layout

```text
.
├── README.md
├── CITATION.cff
├── kaggle_v44/              # TUM notebook, model, data, training and evaluation
├── kaggle_v46/              # CO3D notebook, model, data, training and evaluation
├── tests/                  # Tests for these two implementations
├── tools/check_notebooks.py # Verify notebook cells and embedded source
├── requirements.txt
└── requirements-gpu.txt
```

The Kaggle setup cells install the pinned GPU extras while preserving the
base Torch/torchvision stack. For local use, install a compatible
PyTorch/torchvision CUDA environment together with the listed dependencies.
The requirements files alone do not provision a complete CUDA environment.

```sh
python tools/check_notebooks.py
PYTHONPATH=. python -m pytest -q tests
```

The notebook check needs `nbformat`. The tests additionally require PyTorch;
download fixture tests use local HTTP servers. Training and Gaussian
rendering require the appropriate CUDA environment.

Repository: <https://github.com/Anwarshamim01/multi-view-3d-reconstruction-using-v-jepa-features>
