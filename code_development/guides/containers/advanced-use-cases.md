(use-cases-nris)=
# Use Cases
Some of the use cases where we need to use apptainer are discussed below.

## Using a Custom TensorFlow Version
A common use case on NRIS systems is needing a specific or newer version 
of a software framework (like TensorFlow) than what is globally available 
via the module system.

### The Scenario
A user visits the official TensorFlow installation guide and finds the 
standard Docker commands:
```bash
# Standard Docker commands (Not available on HPC compute clusters)
docker pull tensorflow/tensorflow:latest
docker run -it tensorflow/tensorflow:latest python
```
Since Docker requires root privileges and is not available on our clusters,
it is possible to use Apptainer to accomplish the exact same task.

### Step 1: Pull the Docker Image on the Login Node
You can pull images directly from Docker Hub using the `docker://` prefix. 
Always do this on a login node before submitting or starting any jobs.

```bash
[SAGA]$ apptainer pull --name tensorflow_latest.sif docker://tensorflow/tensorflow:latest
```

### Step 2: Test the Image on a Compute Node
Because importing heavy machine learning libraries like TensorFlow demands a 
lot of CPU power and memory, never run tests on the shared login nodes. Doing so will 
often cause the system to automatically kill your process.
Instead, start a quick interactive compute session:

```bash
[SAGA]$ srun --account=<myaccount> --mem-per-cpu=4G --time=00:30:00 --pty bash
```

### Step 3: Run the Verification Command
```{note}
By default, Apptainer mounts your `$HOME` directory. If you have any older Python 
packages installed locally in your cluster account `(under ~/.local/)`, the container 
might accidentally import them and crash due to a version mismatch.

To ensure the container only uses its own pristine, internal libraries, always 
include the `--env PYTHONNOUSERSITE=1` flag and use exec to run your command:
```
Once you are on a compute node, check the TensorFlow version.

```bash
[saga ~]$ apptainer exec --env PYTHONNOUSERSITE=1 tensorflow_latest.sif python -c "import tensorflow as tf; print(tf.__version__)"
```

## Accessing Older Operating System and Legacy Libraries

Another powerful feature of containers is the ability to run legacy software 
that requires an older operating system version or specific language runtimes 
(like older Perl or Python environments) that are no longer supported on our 
host cluster.

### The Scenario
A user needs to run a legacy bioinformatics pipeline built around BioPerl. 
The pipeline expects an older version of Ubuntu and specific Perl core 
configurations that do not match our cluster's current setup.

### Pull the Image on the Login Node
On the login node, we can pull the official stable image from Docker Hub using:

```bash
[saga ~]apptainer pull --name bioperl_stable.sif docker://bioperl/bioperl:stable
```

### Step 2: Verify the Isolated OS Environment (On a Compute Node)

After launching an interactive session (srun), we can verify that the container 
successfully brings its own unique ecosystem by querying its operating system 
release file:

```bash
[saga ~] apptainer exec bioperl_stable.sif cat /etc/os-release
``` 

Expected output:
```bash
NAME="Ubuntu"
VERSION="14.04.5 LTS, Trusty Tahr"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 14.04.5 LTS"
VERSION_ID="14.04"
```
This shows that even though our cluster host runs a modern enterprise Linux 
distribution, inside this container environment, we are executing commands 
natively within an isolated Ubuntu 14.04 space (released in 2014).

### Executing the Legacy Software
You can seamlessly run code inside the containerized langauge runtime by 
using `apptainer exec` to force the container to run the Perl command. 
We will inspect the `%INC` variable, which tracks exactly where Perl is 
loading its libraries from.

```bash
[saga ~]$ apptainer exec --env LC_ALL=C bioperl_stable.sif perl -e 'use Bio::SeqIO; print join "\n", %INC; print "\n"'
```


```{note}
When running this, you may see a minor warning stating 
`perl: warning: Setting locale failed`. This happens because the container 
is stripped down and missing local language packs. You can safely cleanly 
suppress it by passing `LC_ALL=C` via the `--env` flag.
```

Expected Output:
```bash
parent.pm
/usr/share/perl/5.18/parent.pm
Bio/SeqIO.pm
/usr/local/share/perl/5.18.2/Bio/SeqIO.pm
Bio/Root/Root.pm
/usr/local/share/perl/5.18.2/Bio/Root/Root.pm
base.pm
/usr/share/perl/5.18/base.pm
... [truncated for readability] ...
```

The paths `/usr/share/perl/5.18/` and `/usr/local/share/perl/5.18.2/` exist 
only inside the container's isolated filesystem. Hence, it shows that 
the container is successfully using its own internal Perl version `5.18.2`, 
completely ignoring whatever  version of Perl is installed natively 
on the Saga host. The legacy scripts can now utilize these specific older 
library dependencies without any compilation conflicts.
