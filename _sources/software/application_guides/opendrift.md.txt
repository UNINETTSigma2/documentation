# OpenDrift

This recipe for a **containerized** [OpenDrift](https://opendrift.github.io/)
was provided by a user and is hopefully useful for others.

In a project folder (the home folder is likely not large enough), run the
following:
```console
$ singularity pull docker://opendrift/opendrift
```

This downloads a large file (`opendrift_latest.sif`) which provides OpenDrift
in a container image.

Then create a Python script which imports opendrift:
```python
import opendrift

print("the import worked well")
```

This script can then be run using:
```console
$ ./opendrift_latest.sif python myscript.py
```

It is also possible to open python and run OpenDrift interactively using:
```console
$ ./opendrift_latest.sif python
```

For this to work, you might have to mount specific catalogues (for example
where the ocean model forcing files are) using `SINGULARITY_BIND`:
```console
$ export SINGULARITY_BIND="/cluster"
```

If more directories are needed, they can be added through:
```console
$ export SINGULARITY_BIND="/cluster,/opt,/data"
```
