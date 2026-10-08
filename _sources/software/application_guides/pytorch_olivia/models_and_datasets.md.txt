(pytorch-models-datasets)=

# Models, Datasets, Caches, and Overlays on Olivia

This page summarizes recommended defaults for storing models, datasets, Hugging Face caches, and overlay images on Olivia.

Use it together with {ref}`access-pytorch` and {ref}`pytorch-overlay-images`.

## Recommended Defaults

1. Use the **NRIS Module** by default.
2. Use the **Container Implementation** when you need explicit control over container launch details and want to add extra package.
3. Do **not** plan around extending **EESSI**  and **NRIS PyTorch module** with `pip install`.
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

If additional Python packages are needed, prefer the **direct container approach**.

For project work, the recommended default is:

1. Create one overlay per project.
2. Build it from a `requirements.txt` file.
3. Store it in the project area.
4. Reuse it across related jobs.

```{note}
The package-install workflow is documented in {ref}`pytorch-overlay-images`. This page only describes the recommended organization.
```

## Hugging Face and Torch Cache Locations
PyTorch and Hugging Face workflows often download model weights, datasets, and cache files automatically.

For our project, the datasets `CIFAR-100` and `Tiny-ImageNet` are pre-fetched into the `datasets/` directory inside our project root using the `download_datasets.sh` shell script. `dataset_utils.py` then manages loading and runtime extraction directly from this location, avoiding downloads inside the home directory. 

Only, in the case of traning with ViT model, the pretrained weights from `torchvision` are cached under the Torch cache path (by default often `~/.cache/torch`). If you want all model artifacts to stay inside project storage, set `TORCH_HOME` in the job script. In our example, Hugging Face cache variables are optional because we do not use Hugging Face datasets/models. However, if you decide to use it for your project, please follow these guidelines for it:

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

1. **`HF_HOME`** base Hugging Face cache root.
2. **`HF_HUB_CACHE`** Hugging Face Hub model/download cache.
3. **`HF_DATASETS_CACHE`** Hugging Face Datasets cache.
4. **`TRANSFORMERS_CACHE`** Transformers model cache.
5. **`TORCH_HOME`** Torch/Torchvision model artifact cache (for example pretrained checkpoints).

## Where to Put Models and Datasets

1. For personal work, store models and datasets under your own project or work directory.
2. For shared project work, store them in a shared project location.
3. Apply the same rule to datasets downloaded from outside Hugging Face.
4. Do not let large model and dataset caches build up in the home directory.
