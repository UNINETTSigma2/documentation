(pytorch-models-datasets)=

# Models, Datasets, Caches, and Overlays on Olivia

This page summarizes recommended defaults for storing models, datasets, Hugging Face caches, and overlay images on Olivia.

Use it together with {ref}`access-pytorch` and {ref}`pytorch-overlay-images`.

## Recommended Defaults

1. Use the **module path** by default.
2. Use the **direct container path** when you need explicit control over container launch details.
3. Do **not** plan around extending **EESSI** with `pip install`.
4. Store models, datasets, caches, and overlays in **project or work storage**, not in your home directory.
5. Use **one overlay per project**, not one overlay per job.
6. Build overlays from a `requirements.txt` file and reuse them across related jobs.
7. If several users need the same models or datasets, use a shared project location.

```{note}
Home storage is limited by default, so large model and dataset caches should not be allowed to accumulate there.
```

## Recommended Layout

A reasonable default layout is:

```text
/cluster/work/projects/<project>/<user>/my_project/
├── code/
├── data/
├── hf_cache/
│   ├── hub/
│   ├── datasets/
│   └── torch/
└── overlays/
    └── project_overlay.img
```

If several users in the same project need access to the same models or datasets, place shared caches and overlays in a project-shared location instead.

## Overlay Recommendation

If additional Python packages are needed, prefer the **module path** or the **direct container path**.

For project work, the recommended default is:

1. Create one overlay per project.
2. Build it from a `requirements.txt` file.
3. Store it in the project area.
4. Reuse it across related jobs.

```{note}
The package-install workflow is documented separately in {ref}`pytorch-overlay-images`. This page only describes the recommended organization.
```

## Hugging Face and Torch Cache Locations

PyTorch and Hugging Face workflows often download model weights, datasets, and cache files automatically.

On Olivia, redirect those caches away from your home directory.

The PyTorch guide examples use the following pattern:

```bash
HF_ROOT="${SCRIPT_DIR}/hf_cache"
mkdir -p "${HF_ROOT}/hub" "${HF_ROOT}/datasets" "${HF_ROOT}/torch"

export HF_HOME="${HF_ROOT}"
export HF_HUB_CACHE="${HF_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_ROOT}/hub"
export TORCH_HOME="${HF_ROOT}/torch"
```

These variables control:

1. **`HF_HOME`** sets the general Hugging Face home directory.
2. **`HF_HUB_CACHE`** stores downloaded model files from Hugging Face Hub.
3. **`HF_DATASETS_CACHE`** stores datasets handled through Hugging Face Datasets.
4. **`TRANSFORMERS_CACHE`** stores cached model files used by Transformers.
5. **`TORCH_HOME`** stores Torch-related cached files such as downloaded model artifacts.

## Where to Put Models and Datasets

1. For personal work, store models and datasets under your own project or work directory.
2. For shared project work, store them in a shared project location.
3. Apply the same rule to datasets downloaded from outside Hugging Face.
4. Do not let large model and dataset caches build up in the home directory.
