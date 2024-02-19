(overlay)=
# Creating a Singularity Overlay with Custom Python Packages on LUMI-G HPC

This guide is tailored for users of the LUMI-G High-Performance Computing (HPC) system who need to augment the prebuilt Singularity images provided by LUMI-G with additional Python packages. By creating a Singularity overlay, users can install custom packages without altering the original image, thus maintaining a consistent and reproducible environment.

## Prerequisites

- Access to the LUMI-G HPC system
- A Singularity image (referenced by the `$SIF` environment variable) provided by LUMI-G
- A `requirements.txt` file listing the desired Python packages


## Step 1: Prepare the `requirements.txt` File

Create a `requirements.txt` file that lists all the Python packages you need to add to the LUMI-G provided Singularity image. This file will be used to specify the dependencies that are not already included in the prebuilt image.

Example `requirements.txt`:

```
wandb
opencv-python
torchmetrics
albumentations
einops
pandas
matplotlib
```

## Step 2: Create the `install.sh` Script

The `install.sh` script automates the installation of packages listed in `requirements.txt`.
It sets up the environment and installs the packages into a staging directory that will later be converted into an overlay.

Script `install.sh`:

```bash
$WITH_CONDA

set -xeuo pipefail

export TMPDIR=$PWD

export INSTALLDIR=$PWD/staging/$(dirname $(dirname $(which python)) | cut -c2-)
export PYMAJMIN=$(python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
export PYTHONPATH=$INSTALLDIR/lib/python$PYMAJMIN/site-packages

pip install --no-cache-dir --prefix=$INSTALLDIR --no-build-isolation -r requirements.txt

exit
```

Make the script executable:

```bash
chmod u+x install.sh
```

## Step 3: Create the `create_overlay.sh` Script

The `create_overlay.sh` script facilitates the creation of a squashfs overlay file from the staging directory by running the `install.sh` script within the Singularity image environment. This overlay file encapsulates the additional Python packages.

Script `create_overlay.sh`:

```bash
singularity exec -B$PWD $SIF ./install.sh 1>&2
chmod -R 777 staging/
mksquashfs staging/ overlay.squashfs -no-xattrs -processors 8 1>&2
rm -rf staging
```

Make this script executable:

```bash
chmod u+x create_overlay.sh
```

## Step 4: Execute the Overlay Creation Script

Execute the `create_overlay.sh` script to produce the `overlay.squashfs` file. This file represents the custom environment additions and can be transported and used across the LUMI-G system.

```bash
./create_overlay.sh
```

## Step 5: Update Your Singularity Command

When running your Singularity container on LUMI-G, append the `--overlay=overlay.squashfs` option to your command. This action integrates the overlay with the prebuilt image, allowing you to use the additional packages seamlessly.

Example Singularity command with overlay:

```bash
singularity exec --overlay=overlay.squashfs $SIF <your_command_here>
```
