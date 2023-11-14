WIP





# ParaView using X11 Forwarding and Apptainer (singularity)

In this guide, we are going to use containers provided by OpenFOAM Foundation: https://hub.docker.com/u/openfoam

**NOTE**: We are going to use `$USERWORK` for space purposes. Please, remember that this folder is subject to automatic clean up: https://documentation.sigma2.no/files_storage/clusters.html#user-work-area . It might be necessary to download the container again at some point in the future (which will be available as long as Nvidia maintains it) but, DO NOT store important data under this directory.


## Pulling ParaView image

First, log in to your preferred server via SSH (in this example, we are going to use Fram): `ssh -X -Y username@fram.sigma2.no`

The first time, you will have to pull the container image, and since these can be quite large it is often better not to use your $HOME but the $USERWORK instead: `cd $USERWORK`

Also, let's set some variables so there won't be issues while downloading the container:<br>
`mkdir -p $USERWORK/.apptainer`<br>
`export APPTAINER_CACHEDIR=$USERWORK/.apptainer`<br>
`export APPTAINER_TMPDIR=$USERWORK/.apptainer`<br>

Now, pull a ParaView container image with Apptainer: `apptainer pull docker://openfoam/openfoam-dev-paraview510`

This will create a `.sif` file in the directory from where you pulled it (you can rename this file as much as you want, and also move it where you want, it will still work): `ls -lrtah paraview*`


## X Server for running the application

You first need to download an X server so the GUI can be forwarded and you can interact with ParaView.<br>
For Windows, you can use Xming or VcXsrv. If you use the latter, select "One large window", "Start no client", uncheck "Native opengl" and check "Disable access control"<br>
For Mac, you can use XQuartz<br>

More information here: https://documentation.sigma2.no/getting_started/ssh.html#x11-forwarding and here: https://documentation.sigma2.no/jobs/interactive_jobs.html#graphical-user-interface-in-interactive-jobs

Make sure the application is running and it says, when you hover the mouse over it: "nameOfTheMachine:0.0"


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


## Running the container

Run the following commands:
1. `cd $USERWORK`
2. `singularity shell openfoam-dev-paraview510_latest.sif`
3. `cd /opt/paraviewopenfoam510/bin`



