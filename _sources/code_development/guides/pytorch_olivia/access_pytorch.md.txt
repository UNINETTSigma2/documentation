(access-pytorch)=

# How to Use PyTorch on Olivia
Olivia provides multiple ways to leverage the PyTorch framework for your machine learning and deep learning projects. Below are the two primary methods:
1. **Using the EESSI software stack** 
2. **Using the Container Solution**

## Using EESSI software stack for PyTorch
The **EESSI (European Environment for Scientific Software Installations)** is a cutting-edge service that provides optimized scientific software installations on any machine, anywhere in the world, almost instantly. This eliminates the hassle of manually building or installing software. To learn more about EESSI, visit the [official EESSI website](https://eessi.github.io/). On Olivia, the EESSI software stack is pre-configured and can be activated using the following command:

```bash
module load EESSI/2025.06
```
Once the EESSI stack is activated, you can load PyTorch as a module using:

```bash
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
```

If your project involves computer vision tasks, you can also load the torchvision module:
```bash
module load torchvision/0.22.0-foss-2024a-CUDA-12.6.0
```

To explore additional packages available in the EESSI stack, use the `module spider` command. This will help you search for any required modules for your PyTorch project. Most commonly used packages should already be available. However, if you find any missing packages, feel free to reach out to us, and we’ll ensure they are installed for you.


## Using the Container Solution
Container are the most preferred solution for many of the users. Information about the pre-built container, downloading one and then using it is available here.({ref}`olivia_pilot_main`). Moreover, you can find the information about how to use container for PyTorch in your deeplearning project referring to this documentation ({ref}`pytorch-on-olivia`). It shows the utilization of containers for training your deep learning project on single gpu and also scaling it to multiple GPUs and nodes.


When working with containerized environments, you may encounter situations where the base container does not include all the software or Python packages required for your workflow. Instead of modifying the container itself, you can extend it in two ways:
1. **Using SquashFS** : 

SquashFS is a compressed, read-only file system that can be mounted alongside the container. This allows you to add custom environments or dependencies without altering the original container. It is an efficient way to include additional software or packages while keeping the container immutable.

2. **Using Overlay File Systems** :

 Overlay file systems provide a writable layer on top of the container, enabling you to add or modify files dynamically during runtime. This could be taken as a writable layer placed on top of your container.When you extend the container, the system looks at the overlay first and if the file isn´t there, then it looks at the base `.sif` image.


### Steps to extend a Container Using SquashFS
We will see how we can create and use a SquashFS file to extend a containerized environment. The process involves creating a Python virtual environment, installing additional dependencies, and packaging the environment into a SquashFS file.

1. **Set Up the Container**

First, ensure you have access to the container you want to extend. In this example, we use a container named pytorch_25.05-py3.sif. We will use this container to execute all the commands and create the extended environment.

```bash
export SIF=./pytorch_25.05-py3.sif
```


2. **Create a Python Virtual Environment**

To add custom Python packages, start by creating a virtual environment inside the container.This ensures that the environmen is isolated and can be package later. Please note that you need to bind  `--bind $PWD`  the current working directory to the container which will ensure that all paths inside the container resolve correctly to the host system´s path.

```bash
apptainer exec --bind $PWD $SIF bash -c "python -m venv myenv --system-site-packages
```


3. **Install Additional Packages**

Next, install the required Python packages into the virtual environment. These packages are typically listed in `requirements_additions.txt` file. For instance, there could be there packages needed for your PyTorch project missing in the container.

- requirements_addition.txt
````bash
wandb
psutil
pillow
imageio
imgui
glfw
PyOpenGL
imageio-ffmpeg
pyspng
lmdb
dcor
````
To activate the virtual environment and then install the packages in the list, please use this command.

```bash
apptainer exec --bind $PWD $SIF bash -c "
  source myenv/bin/activate && \
  pip install -q -r requirements_additions.txt
```


4. **Test the Environment**

Before packaging the environment, we can test it to ensure all the dependencies are installed correctly.For instance, you can run a Python Script `test_imports.py` to verify those imports.

```bash
apptainer exec --bind $PWD $SIF bash -c "source myenv/bin/activate && python test_imports.py
```

- test_imports.py
```python
import importlib
# List of packages to test
packages = [
    "filelock",
    "fsspec",
    "jinja2",
    "markupsafe",
    "typing_extensions",
    "numpy",
    "click",
    "PIL",  # Pillow is imported as "Pillow" but the package name is "pillow"
    "scipy",
    "requests",
    "tqdm",
    "ninja",
    "matplotlib",
    "imageio",
    "torch",
    "torchvision",
    "imgui",
    "glfw",
    "OpenGL",  # PyOpenGL is imported as "OpenGL"
    "imageio_ffmpeg",  # imageio-ffmpeg is imported as "imageio_ffmpeg"
    "pyspng",
    "wandb",
    "pydantic",
    "tensorboard",
    "lmdb",
    "dcor",
    "psutil",
]
# Test imports
def test_imports():
    print("Testing package imports...\n")
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"Successfully imported {package}")
        except ImportError:
            print(f"Failed to import {package}")
    print("\nImport test completed.")
if __name__ == "__main__":
    test_imports()
```

5. **Create the SquashFS File**

Once the environment is ready, package it into a SquashFS file. This file will contain the entire virtual environment in a compressed, read-only format.`mksquashfs` creates a SquashFS file name `packages.sqsh` from the `myenv` directory. `-noappend` ensures that the file is created from scratch and `-no-progress` suppresses progress output.

```bash
mksquashfs myenv packages.sqsh -noappend -no-progress
```


6. **Clean Up the Virtual Environment**

After creating the SquashFS file, we can delete the original virtual environment directory to save space. This is major benefit of SquashFS file system as it will not impact the filesystem.

```bash
rm -rf myenv
```


7. **Test the SquashFS Environment**

Finally, after removing the virtual environment directory, the next step is to test it by mounting it alongside the container and verifying that the environment works as expected which could be done by running the python script `test_imports.py` inside the container.

The SquashFS file `packages.sqsh` is mounted inside the container using the `-B` option in the `apptainer exec` command. Specifically `-B packages.sqsh:/user-software:image-src=/`. This commands mounts the SquashFS file `packages.sqsh` as the directory `/user-software` inside the container which allows the container to access the contents of SquashFS file as they were part of the container´s filesystem.And the environment stored in the SquashFS file can be then activated using `source /user-software/bin/activate`.

````bash
export APPTAINERENV_PREPEND_PATH=/user-software/bin
apptainer exec -B $PWD -B r3gan-env.sqsh:/user-software:image-src=/ $SIF bash -c "source /user-software/bin/activate && python test_imports.py"
````

The `APPTAINERENV_PREPEND_PATH` variable allows you to prepend a directory (`/user-software/bin` as shown above) to the container´s `PATH` environment variable. This ensures that the binaries and executables located in `/user-software/bin` such as the Python interpreter take precedence over other binaries in the container.Hence, by preprending it, the container will prioritize using the Python evnironment and tools from the SquashFS file over the default system Python or other executables  in the container.

### Steps to extend a Container Using Overlay
1. **Create the Overlay file**

First we need to create a blank, formatted image file that will hold your additional packages. Note that, without the `--sparse` option, the overlay file will immediately consume the full 5GB of disk space even if it is empty.So, specifying it will only consume as much disk space as is actually needed to store data.

````bash
export OVERLAY=my_overlay.img
apptainer overlay create --sparse --size 5120 $OVERLAY
````

2. **Install packages into the Overlay**

To install your additional packages into this layer, you can run the container with the `--overlay` flag and use `pip`. Using the `--user` flag inside the overlay ensures that the files are written to the writable layer, not the read-only container system paths.

```bash
# We use --overlay to mount the writable layer
# We install to a local path inside the overlay to avoid permission issues
apptainer exec --overlay $OVERLAY --bind $PWD $SIF bash -c "
    export PYTHONUSERBASE=/home/apptainer/user_software
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    pip install --user --no-cache-dir -r requirements_additions.txt
"
```

3. Run your project with the Overlay
Once installed, you can simply include the overlay flag everytime you run your code.

```bash
# You must include the overlay and set the paths so Python finds the packages
apptainer exec --overlay $OVERLAY --bind $PWD $SIF bash -c "
    export PYTHONUSERBASE=/home/apptainer/user_software
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH
    python test_imports.py
```



Notes:
1. `PYTHONUSERBASE`: Since you cannot write to the root of the container, we define a directory inside the overlay `(/home/apptainer/user_software)` to hold those libraries.
2. `pip install --user`: This tells `pip` to install packages into the path defined by `PYTHONUSERBASE` instead of system folders.
3. `apptainer overlay create`: It creates a physical file that act as a writable disk. So, we dont have to crete a directory or virtual environment.

Below is the full job script that will allow you to create an Overlay image for your extended software requirements.

```bash
#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name=create_overlay
#SBATCH --output=create_overlay_%j.out

echo "========================================"
echo "Creating Apptainer Overlay Environment"
echo "Started: $(date)"
echo "========================================"

# Set container and overlay paths
export SIF=./pytorch_25.05-py3.sif
export OVERLAY=my_overlay.img

echo ""
echo "=== Step 1: Create a 5GB Sparse Overlay File ==="
# Sparse files take up almost no space until you actually fill them
apptainer overlay create --sparse --size 5120 $OVERLAY

echo ""
echo "=== Step 2: Install packages into the Overlay ==="
# We use --overlay to mount the writable layer
# We install to a local path inside the overlay to avoid permission issues
apptainer exec --overlay $OVERLAY --bind $PWD $SIF bash -c "
    export PYTHONUSERBASE=/home/apptainer/user_software
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    pip install --user --no-cache-dir -r requirements_additions.txt
"

echo ""
echo "=== Step 3: Test the Overlay environment ==="
# You must include the overlay and set the paths so Python finds the packages
apptainer exec --overlay $OVERLAY --bind $PWD $SIF bash -c "
    export PYTHONUSERBASE=/home/apptainer/user_software
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH
    python test_imports.py
"

echo ""
echo "========================================"
echo "✓ SUCCESS!"
echo "Overlay created: $OVERLAY"
echo "Completed: $(date)"
echo "========================================"
```
