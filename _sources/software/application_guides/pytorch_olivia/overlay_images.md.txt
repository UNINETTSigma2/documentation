(pytorch-overlay-images)=

# Adding Python Packages to PyTorch Containers

Base container images rarely ship with every dependency you need for a project. In our case, the core PyTorch environment was ready to go, but we still needed `wandb` for experiment tracking. Instead of rebuilding the entire container, we use a [Persistent Overlays](https://apptainer.org/docs/user/latest/persistent_overlays.html). An overlay acts as a lightweight, writable layer on top of an otherwise immutable SIF image. When you install packages or write files inside the container, Apptainer captures those modifications in the overlay file so they remain available for every future training run.

We recommend installing packages from a `requirements.txt` file in a short Slurm job with `#SBATCH --gpus-per-node=0`, since package installation does not need GPU resources.

Below, we'll demonstrate how to extend your container using an overlay.

`````{tabs}

````{group-tab} Pip Packages With Containers

First, create a `requirement.txt` file containing the packages you need. In this case, `wandb`.
```bash
wandb
```

Next, save the following Python script as `test_imports.py`.

```python
import sys
import wandb

print("========================================")
print("     OVERLAY PACKAGE VERIFICATION       ")
print("========================================")

print(f"Python Executable : {sys.executable}")
print("\nActive Python Path (sys.path):")
for idx, path in enumerate(sys.path):
    print(f"  [{idx}] {path}")

print("\n--- Package Resolution Check ---")
print(f"wandb location    : {wandb.__file__}")
print("========================================")
print("Verification complete! 'wandb' is available from overlay.")
print("========================================")
```

With your project files ready, you can submit the following Slurm script `build-overlay.sh` to build the overlay using `sbatch build-overlay.sh`, install wandb, and verify the installation.

```bash
#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --partition=accel
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --job-name=o_user
#SBATCH --output=ouser_%j.out

# Configuration
export SIF=/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif
export OVERLAY=overlay.img
export OVERLAY_DIR=/home/apptainer/work/user_packages

# Direct overlay redirection via APPTAINERENV_
export APPTAINERENV_PYTHONUSERBASE=$OVERLAY_DIR

echo "=== Step 1: Create Sparse Overlay ==="
apptainer overlay create --sparse --size 2048 $OVERLAY

echo "=== Step 2: Install Package into Overlay ==="
apptainer exec --overlay $OVERLAY $SIF bash -c "pip install --user --no-cache-dir -r requirement.txt"

echo "=== Step 3: Verify Overlay Import ==="
apptainer exec --overlay $OVERLAY $SIF bash -c "python test_imports.py"
```

In the script above, we first specify the base container and define the output overlay image `overlay.img`. We then set up the target installation directory inside the container `/home/apptainer/work/user_packages`.

When creating the overlay, the `--sparse` flag is essential. It prevents allocating the full 2GB upfront, allowing the file size to grow only as packages are installed. To install dependencies from `requirements.txt`, wrap your inner commands in `bash -c "..."`, so Apptainer reliably invokes the container’s `pip` and `python` binaries. Once created, attach the overlay to any future run using `--overlay $OVERLAY`.

The logs below demonstrate the resulting installation directory and package import search order.

```bash
========================================
     OVERLAY PACKAGE VERIFICATION
========================================
Python Executable : /usr/bin/python

Active Python Path (sys.path):
  [0] /cluster/work/support/<user_name>/pytorch_olivia/overlay
  [1] /usr/lib/python312.zip
  [2] /usr/lib/python3.12
  [3] /usr/lib/python3.12/lib-dynload
  [4] /home/apptainer/work/user_packages/lib/python3.12/site-packages
  [5] /usr/local/lib/python3.12/dist-packages
  [6] /usr/local/lib/python3.12/dist-packages/nvfuser-0.2.27a0+9bf5aca-py3.12-linux-aarch64.egg
  [7] /usr/local/lib/python3.12/dist-packages/lightning_thunder-0.2.3.dev0-py3.12.egg
  ....
  [13] /usr/lib/python3/dist-packages

--- Package Resolution Check ---
wandb location    : /home/apptainer/work/user_packages/lib/python3.12/site-packages/wandb/__init__.py
========================================
Verification complete! 'wandb' is available from overlay.
========================================
```
Looking at the logs above, Python inserts the `PYTHONUSERBASE` path at index `4` in `sys.path`. Because this appears before system site-packages like `/usr/local/lib/python3.12/dist-packages`, any packages installed into the overlay via `--user` will be loaded first, cleanly overriding base container versions.

The output confirms that wandb is installed under `/home/apptainer/...` inside the `ext3` overlay image rather than on the host filesystem. This distinction is vital on shared clusters. A typical `Python` and `Conda` setups generate thousands of small files that degrade performance on  parallel filesystems like Lustre. Encapsulating these dependencies within an overlay presents them to the storage system as a single file, protecting cluster performance.

```{note}
1. **CLI Commands** : Python modules import automatically, but executable binaries such as `wandb` CLI commands require adding `APPTAINERENV_PATH=/home/apptainer/work/user_packages/bin:$PATH` to your Slurm script.

2. **Environment Isolation** : Installing packages via `--user` can accidentally upgrade existing base libraries in the container. If your workflow requires strict dependency isolation, a isolated `venv` overlay is recommended.
```
````
`````

## EESSI Path & NRIS PyTorch module

Both EESSI & NRIS PyTorch Module is not intended for this extension model.

If required Python packages are missing, use the direct container implementation instead.
