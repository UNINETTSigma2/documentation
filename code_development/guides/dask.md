---
orphan: true
---

(dask-tutorial)=

# Using Dask to scale your Python program
Dask is a Python library that allows you to scale your existing Python code for optimal use on HPC systems. More information on Dask can be found [here](https://dask.org/). Here, we demostrate a simple example of how the Dask `delayed` function can be used to parallelize your code and how to create a Dask cluster using the `SLURMCluster`.

```{contents} Table of Contents
```
## Parallelize your code with dask.delayed
This example is adapted from the Dask documentation on dask.delayed found [here](https://docs.dask.org/en/stable/delayed.html).

Imagine you have a large amount of data that needs to be processed before you can do any analysis. If the data can be separated into smaller chuncks that can be processed independently, you can use the `dask.delayed` function to parallelize your code. In this example we consider such a scenario where we have a list of data (`all_data`) that can be processed independently. Using a for-loop, we process all the data independently using the function `increment_data`. Once the for-loop is complete, we do the final analysis using the function `do_analysis`.
```{code-block} python
def increment_data(data):
    return data + 1

def do_analysis(result):
    return sum(result)

all_data = [1, 2, 3, 4, 5]
results = []
for data in all_data:
    data_incremented = increment_data(data)
    results.append(data_incremented)

analysis = do_analysis(results)
```

We will now parallize this code by using `dask.delayed` as a decorator to turn the function `increment_data` into a delayed function. The function will now behave *lazy* and return a `Delayed` object instead of the actual result. The `Delayed` object holds your function and its arguments in order to run it in parallel later using a Dask cluster. The actual computation is delayed until the `compute` method is called, here done by calling `analysis.compute()`. 

```{code-block} python
---
emphasize-lines: 1, 3, 16, 17, 18
---
import dask

@dask.delayed
def increment_data(data):
    return data + 1

def do_analysis(result):
    return sum(result)

all_data = [1, 2, 3, 4, 5]
results = []
for data in all_data:
    data_incremented = increment_data(data)
    results.append(data_incremented)

analysis = dask.delayed(do_analysis)(results) # dask.delayed can also be used in 
                                              # in this manner 
final_result = analysis.compute() # execute all delayed functions
```

Next, we will use the `SLURMCluster` class from the `dask-jobqueue` package. This class is used to create a Dask cluster by deploying Slurm jobs as Dask workers. The class takes arguments needed to queue a *single* Slurm job/worker, not the characteristics of your computaion as a whole. The arguments are similar to the #SBATCH commands in a Slurm script. Using the `scale` method, we can scale the cluster to the desired number of workers.

```{code-block} python
---
emphasize-lines: 5, 6, 7,8,9,10,12, 14, 32, 33
---
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory="500M",
                       walltime="00:05:00",
                       project="nn9999k", # replace with your project number
                       interface='ib0')

cluster.scale(5) # scale cluster to 5 workers

client = Client(cluster) # connect to the cluster

@dask.delayed
def increment_data(data):
    return data + 1

def do_analysis(result):
    return sum(result)

all_data = [1, 2, 3, 4, 5]
results = []
for data in all_data:
    data_incremented = increment_data(data)
    results.append(data_incremented)

analysis = dask.delayed(do_analysis(results)) 
final_result = analysis.compute()

cluster.close() # shutdown the cluster
client.close() # shutdown the client
```

Here, we configurated each worker to have 1 core, 500 MB of memory and a walltime of 5 minutes. Using `cluster.scale(5)`, we scaled the the cluster to contain 5 workers. running `squeue -u "username"` after executing your main Slurm script will show that 5 additional Slurm jobs that were created. The figure below shows the task graph created by Dask for this specific Python example. 
![task graph using DASK](dask_taskgraph.png)

## Executing your Python script on the HPC system
First, find available versions of Dask on the HPC system:
```console
$ module spider dask
```
Then load the module for the version you want to use:
```console
$ module load dask/your_version
```

An example Slurm job script is found below. Remember to replace the Slurm parameters with your own.

```{code-block} bash
---
emphasize-lines: 15
---
#!/bin/bash

#SBATCH --account=nn9999k 
#SBATCH --job-name=dask_example
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-00:10:00 

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

# Loading Software modules
module --quiet purge            # Restore loaded modules to the default
module load dask/your_version
module list

python dask_example.py
exit 0
```
Installing Dask in a virtual environments enables you to use versions of Dask not installed globally and to install optional dependencies for Dask. See more information in the following section on {ref}`visualizing-the-task-graph`.


```{note}
It is possible to manually configure a distributed Dask cluster without using the `SLURMCluster` class. This is more advance use of Dask and is not covered in this tutorial. The benefit of this approach is that workers are created inside a main job instead of spawning individual Slurm jobs, thus your calucations are confinded to one Slurm job. More information can be found [here](https://docs.dask.org/en/stable/deploying-python-advanced.html).
```

(visualizing-the-task-graph)=
## Visualizing the task graph

Dask provides the function (`visualize`) which prodcues an image of the task graph. However to use the function, you need to install optional dependencies for Dask, and the easiest way is to create a new virtual environment and install Dask and the optional dependencies here.

### Graphviz

You need both the Graphviz system library *and* the Graphviz Python library installed. The Graphviz system library is installed globally and can be loaded using the `module load` command.

First, find available versions of graphviz on the HPC system:
```console
$ module spider graphviz 
```
Then load the module for the version you want to use:
```console
$ module load graphviz/your_version
```
Now you can create a virtual environment and install Dask, Graphviz and, if you are using `SLURMCluster`, Dask-jobqueue. Here, we will use `pip`. More information about virtual environments and installing Python packages can be found here: {ref}`installing-python-packages`. 
```console
$ python -m venv my_new_pythonenv
$ source my_new_pythonenv/bin/activate
$ python -m pip install dask graphviz dask-jobqueue
```

```{warning}
If you are using a virtual environment, you need to make sure that the virtual environment is created with the same Python version that the Graphviz module uses. For example, if you are using Graphviz/2.47.2-GCCcore-10.3.0, you need to create the virtual environment with Python 3.9.5. To find the correct Python version, load Graphviz with `module load graphviz/your_version` then run `module list` and look for the Python version which was just loaded with Graphviz.
```

### Slurm job script using virtual environment

```{code-block} bash
---
emphasize-lines: 14, 16, 17, 18
---
#!/bin/bash

#SBATCH --account=nn9999k 
#SBATCH --job-name=dask_example
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-00:10:00 

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module --quiet purge            
module load Graphviz/your_version # replace with the version you want to use

export PS1=\$
source my_new_pythonenv/bin/activate # replace my_new_pythonenv with the name 
                                     # of your virtual environment

# It is also recommended to to list loaded modules, for easier debugging
module list

python dask_example.py
exit 0
```

You can now safely include the `visualize` function in your script.

```{code-block} python
---
emphasize-lines: 9, 10, 32
---
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory="500M",
                       walltime="00:05:00",
                       account="nn9999k", # NB: in newer versions of dask-jobqueue, "project"
                                          # has been renamed to "account"
                       interface='ib0')
cluster.scale(5)

client = Client(cluster)

@dask.delayed
def increment_data(data):
    return data + 1

def do_analysis(result):
    return sum(result)

all_data = [1, 2, 3, 4, 5]
results = []
for data in all_data:
    data_incremented = increment_data(data)
    results.append(data_incremented)

analysis = dask.delayed(do_analysis)(results)
final_result = analysis.compute() 

analysis.visualize(filename="visualize_taskgraph.svg")

cluster.close()
client.close()
```
