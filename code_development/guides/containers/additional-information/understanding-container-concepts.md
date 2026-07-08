(understanding-container-concepts)=
# What is a container image
In computing, a container image is a lightweight, standalone 
package that contains everything an application needs to run.
Instead of installing software directly onto a computer and 
risking version conflicts with other apps, an image bundles the 
application code, runtime, system tools, libraries and 
configurations into a single, unchangeable file.

The defining character of a container is that it does not bundle 
a full operating system. Instead, it virtulizes at the software layer, 
sharing the host machine´s Linux kernel while maintaining its own 
completely isolated user space.


## Understanding Container Image vs Running Container

To use containers effectively, it is important to understand the 
distinction between an Image and a Container.
- `The Container Image (The Blueprint)` :
This is a frozen, read-only file (such as a `.sif` file on our cluster). 
It acts as a blueprint and contains all the static software layers, 
libraries and code, but it doesnot do anything on its own.

- `The Running Container(The Instance)` :
This is the active process created when you execute an image
(e.g. via `apptainer shell`). It uses active system CPU and RAM to run 
your workloads. Running containers are transient, when your training 
job finishes or you exit the session, the container disappears leaving 
the original image completely unaltered. 

If you want a different environment, you don´t modify a running container 
but you simply use a different image.

Hence, if you wrap your software stack inside a container, it solves 
the critical challenge of reproducibility. Every software dependecny 
from the base operating system and CUDA drivers,frameworks and specific 
codes is locked down in a clear unchangeable record. Moreover, Containers 
abstract away the underlying host hardware which allows build and test 
workflow on any of these platforms (Windows, MAC  or Linux-based HPC cluster)
 
```{note}
If you see a container image (`.sif`) in a shared directory inside the 
cluster, it is possible to start multiple container instances from it.
Hence, many user can use the exact same file simultaneously. However, each 
user gets their own completely separate, active container instance running on 
different cluster nodes. None of them will interfere with each other and the 
original files remains completely untouched.

```