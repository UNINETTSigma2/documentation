(overlay)=
# Creating a Singularity Overlay with Custom Python Packages on LUMI-G
The LUMI-G High-Performance Computing (HPC) system offers users a collection of
prebuilt Singularity containers, pre-configured with a suite of scientific computing
tools and libraries optimized for the LUMI-G infrastructure. Users looking to add
specific software, such as machine learning frameworks, can follow detailed
installation instructions provided by LUMI, exemplified by the guides for
[PyTorch](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/) and
[TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/).
These guides demonstrate the process of installing additional software stacks using
EasyBuild recipes. After installation, users can activate the software by using
the `module load` command, which also sets the `$SIF` environment variable,
indicating the path to the newly installed Singularity Image File (SIF).

While the prebuilt containers on LUMI-G are comprehensive, users may need to install
additional Python packages not already included. The guide below shows how to create
a Singularity overlay, which allows for the installation of these extra packages onto
the existing container. This method keeps the base environment unchanged, ensuring
reproducibility.

In the guide, the `$WITH_CONDA` environment variable is used to activate the Python
environment within the container, a step necessary for installing new Python packages.
This variable is predefined by the container developers for ease of use.

## Prerequisites

- A singularity image installed using EasyBuild on LUMI-G
- A `requirements.txt` file listing the desired Python packages


## Step 1: Prepare the `requirements.txt` File

Create a `requirements.txt` file that lists all the Python packages you need to
add to the LUMI-G provided Singularity image. This file will be used to specify
the dependencies that are not already included in the prebuilt image.

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
It sets up the environment and installs the packages into a staging directory
that will later be converted into an overlay.

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

The `create_overlay.sh` script facilitates the creation of a squashfs overlay file
from the staging directory by running the `install.sh` script within the Singularity
image environment. This overlay file encapsulates the additional Python packages.

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

Execute the `create_overlay.sh` script to produce the `overlay.squashfs` file.
This file represents the custom environment additions and can be transported and
used across the LUMI-G system.

```bash
./create_overlay.sh
```

## Step 5: Modify Your Singularity Run Command with Overlay

When you're ready to run your Singularity container on LUMI-G with the custom overlay,
you'll need to modify your existing run command by appending the `--overlay` flag.
This flag specifies the path to your `overlay.squashfs` file, which contains the
additional Python packages you've installed.

Here is an example of how such a command might look:

```bash
singularity exec --overlay path_to_overlay/overlay.squashfs $SIF bash -c '$WITH_CONDA; python my_script.py'
```

In this example, replace `path_to_overlay/overlay.squashfs` with the actual path to your overlay file, and `my_script.py` with the script or command line you intend to execute within the container. This modification ensures that your custom environment is active during the container's runtime.
