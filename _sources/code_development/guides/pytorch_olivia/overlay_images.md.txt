(pytorch-overlay-images)=

# Adding Python Packages to PyTorch Containers

Both supported approaches use containers. The difference is where you control the container layer:

1. **Modules** hide the container launch behind the PyTorch module interface.
2. **Containers** expose the Apptainer command directly.

In both cases, extra Python packages are installed into an overlay image instead of the base container. We recommend installing packages from a `requirements.txt` file in a short Slurm job with `#SBATCH --gpus-per-node=0`, since package installation does not need GPU resources. Use the tab that matches how you run PyTorch.

`````{tabs}
````{group-tab} Pip Packages With Modules

Use this approach when you load PyTorch with the module system.

The PyTorch module is container-backed, so extra packages are written to an overlay image instead of the base runtime. To avoid storing overlays in home, set `PYTORCH_OVERLAY_FILE` to a project or work path **before** loading the module.

Recommended layout:

```text
my_project/
├── requirements.txt
└── pytorch-overlay.img
```

Recommended Slurm job:

```bash
#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --partition=accel
#SBATCH --mem=16G
#SBATCH --gpus-per-node=0

set -euo pipefail

export PYTORCH_OVERLAY_FILE="${PWD}/pytorch-overlay.img"
export PYTORCH_OVERLAY_SIZE=5120

ml reset
ml load NRIS/GPU
ml load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
ml use /cluster/work/support/temporary_modules
ml load PyTorch/2.8.0

export PYTORCH_OVERLAY_MODE=rw
pip install --user -r requirements.txt
```

Use read-only mode for normal runtime jobs:

```bash
export PYTORCH_OVERLAY_MODE=ro
```

Verify in a separate job:

```bash
python -c "import humanize, boltons; print('overlay verify ok')"
```

```{important}
For module-based install jobs, use `pip` (not `python -m pip`) when `PYTORCH_OVERLAY_MODE=rw`.
```

```{note}
Run writable package-install jobs one at a time per overlay file. Parallel `rw` installs to the same overlay are not supported.
```

Default module behavior (if you do not override `PYTORCH_OVERLAY_FILE`) uses:

```text
$HOME/.PyTorch/<version>/pytorch-overlay.img
```

````

````{group-tab} Pip Packages With Containers

Use this approach when you run the PyTorch container directly with Apptainer.

In this workflow, you manage the overlay explicitly. Place the image next to `requirements.txt`.

Recommended commands:

```bash
export SIF="/path/to/pytorch_container.sif"  # for example: /cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
export OVERLAY="${PWD}/pytorch-overlay.img"

apptainer overlay create --sparse --size 5120 "$OVERLAY"

apptainer exec --overlay "$OVERLAY" --bind "$PWD" "$SIF" bash -lc '
	export PYTHONUSERBASE=/home/apptainer/user_software
	export PATH="$PYTHONUSERBASE/bin:$PATH"
	pip install --user -r requirements.txt
'
```

Reopen read-only for verification:

```bash
apptainer exec --overlay "${OVERLAY}:ro" --bind "$PWD" "$SIF" bash -lc '
	export PYTHONUSERBASE=/home/apptainer/user_software
	export PATH="$PYTHONUSERBASE/bin:$PATH"
	python - <<"PY"
import humanize
import boltons
print("overlay verify ok")
PY
'
```

```{note}
In direct-container tests, both `pip` and `python -m pip` worked.
```

```{note}
Direct package installs (for example `pip install --user natsort==8.4.0`) also worked, but for project work we recommend `requirements.txt` for reproducibility.
```

````
`````

## EESSI Path

EESSI is not intended for this extension model.

If required Python packages are missing from EESSI, use the module path or the direct container path instead.
