# ParaView with Apptainer (singularity)

Running ParaView using remote desktop software on our clusters is far from ideal because it adds an unnecessary layer of virtualization, making the application run slower and taxing the server and users running other programs.

Running ParaView through a container has a few advantages:
- You do not rely on IT support to install a particular version of the software;
- It is possible to run the latest version, as long as the container image is also updated;
- You can specify exactly how much resources you need (including CPUs and also GPUs where available) and they will be allocated to your project;
- It runs much better on your, already familiar, browser;
- There is no need to maintain old software that will probably stop working within time;
- You can use the same container image on different hosts (i.e. what is described here can be adapted for other platforms), and always remain in the exact same software environment.

In this guide, we are going to use containers provided on NVIDIA NGC, which is a hub for GPU-optimized software for deep learning, machine learning, and HPC: https://catalog.ngc.nvidia.com/orgs/nvidia-hpcvis/containers/paraview

**NOTE**: We are going to use `$USERWORK` for space purposes. Please, remember that this folder is subject to automatic clean up: https://documentation.sigma2.no/files_storage/clusters.html#user-work-area . It might be necessary to download the container again at some point in the future (which will be available as long as Nvidia maintains it) but, DO NOT store important data under this directory.


## Pulling ParaView image

First, log in to your preferred server via SSH (in this example, we are going to use Fram): `ssh -i ~/.ssh/ssh_key username@fram.sigma2.no`

The first time, you will have to pull the container image, and since these can be quite large it is often better not to use your $HOME but the $USERWORK instead: `cd $USERWORK`

Also, let's set some variables so there won't be issues while downloading the container:
`mkdir -p $USERWORK/.apptainer`
`export APPTAINER_CACHEDIR=$USERWORK/.apptainer`
`export APPTAINER_TMPDIR=$USERWORK/.apptainer`

Now, pull a ParaView container image with Apptainer: `apptainer pull docker://nvcr.io/nvidia-hpcvis/paraview:egl-py3-5.11.0`

This will create a `.sif` file in the directory from where you pulled it (you can rename this file as much as you want, and also move it where you want, it will still work): `ls -lrtah paraview*`

**WARNING**: If you want to run a different ParaView version, you can do so by replacing the url after "docker://", copying the new one from here: https://catalog.ngc.nvidia.com/orgs/nvidia-hpcvis/containers/paraview/tags. 
However, if you do this, be careful to use the correct PATH for Paraview because for tags `egl-py3-5.9.0` and `egl-py3-5.8.0`, Paraview was installed in `/opt/paraview` whereas for tags `egl-py3-5.11.0`, `glx-5.6.0rc3` and `egl-5.6.0rc` it is installed in `/usr/local/paraview`, so modify the PATH in "[Apptainer exec command](https://documentation.sigma2.no/software/application_guides/paraview.html#running-the-container)" accordingly.
````

## Allocating resources for the project

Log in again to your preferred server and run the following command: `salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --time=00:30:00 --qos=devel --account=nnxxxxk`

Please, note that here we are asking 1 CPU only for 30 minutes in the Devel queue. **If you need more resources and time, adjust the parameters accordingly.**

The output will be similar to this one:

```
salloc: Pending job allocation 5442258
salloc: job 5442258 queued and waiting for resources
salloc: job 5442258 has been allocated resources
salloc: Granted job allocation 5442258
salloc: Waiting for resource configuration
salloc: Nodes c84-5 are ready for job
```

**NOTE**: Remember which node was allocated for the job, it will be needed later. In this case, the allocated node was "c84-5".

## Running the container

Select a **random port number**, say 7412. Also, for this guide, we will assume your data is located in `$USERWORK/data` 

**REMEMBER**: Please, adjust the command accordingly with the chosen port, data directory and the Paraview image you want to use. If you see an error because the port is already in use, select another port number.

`apptainer exec --bind $USERWORK/data:/data $USERWORK/paraview_egl-py3-5.11.0.sif /usr/local/paraview/bin/pvpython /usr/local/paraview/share/paraview-5.11/web/visualizer/server/pvw-visualizer.py --content /usr/local/paraview/share/paraview-5.11/web/visualizer/www --port 7412 --data /data -i 0.0.0.0`

The command above is binding the port and the data folder to the container, so that it can see the information outside of it (by default, a container is relatively isolated from "outside world", meaning we have to specify which folders from the host machine we want to "see" from inside the container). 

The first folder `$USERWORK/data` is only known outside the container and `/data` is only known inside the container, we are binding them together with `--bind $USERWORK/data:/data` but it is **the same folder** therefore changes made in `/data` are actually done to `$USERWORK/data` and hence permanent.

**From a second terminal window**, log in again to the server but, this time, **forwarding** the port you used for the container: `ssh -L 7412:localhost:7412 -i ~/.ssh/ssh_key username@fram.sigma2.no`

Now, forward again the same port from the compute node that you were allocated, run the following: `ssh -L 7412:localhost:7412 c84-5`. **Remember to replace the last part (c84-5) with the allocated node in the beginning**


## Executing ParaView

Finally, on your computer's browser, type the following address (replacing the chosen port): `127.0.0.1:7412`

You should see ParaView window loading on your browser.
