---
orphan: true
---

# Container with build environment

```{note}
To follow this tutorial you need to have root access to a Linux computer
with Singularity installed, e.g. your personal laptop/workstation.
Please follow the installation
[instructions](https://sylabs.io/guides/3.7/user-guide/quick_start.html)
from the Singularity documentation.
```

Sometimes we encounter applications that have system dependencies which are incompatible
with the global environment on the cluster. This can happen for instance if you download
a precompiled binary from an online archive which has been built for a specific OS version
or depends on a system library which is not available, or if you want to compile your own
application with some non-standard dependencies. One way to resolve such issues is to
containerize the appropriate environment and run/compile your application _through_ this
container on the cluster. In the following examples we will demonstrate such a work flow.

## Hello world example

This example demonstrates:
1. how to write a simple Singularity definition file
2. how to install system packages on top of a standard OS base image
3. how to build the container on your laptop
4. how to run commands through the container environment on the cluster

In this example we will create a very simple container environment with a Ubuntu-16.04
operating system and a GNU compiler. We will then use this environment to compile a
simple program on the cluster.

**Writing the definition file**

We start with the following definition file (we call it `example.def`)
```
Bootstrap: docker
From: ubuntu16.04

%post
    apt-get update && apt-get install -y g++
```

This recipe will pull the `ubuntu16.04` image from the [docker](https://hub.docker.com)
registry and install the GNU C++ compiler using the Ubuntu package manager. Any system
package that is available for the base OS can be installed in this way. Other common
`Bootstrap` options include
`library` for the Singularity [Container Library](https://cloud.sylabs.io/library),
`shub` for [Singularity-Hub](https://singularity-hub.org) or
`localimage` if you want to build on top of another image located on your computer.

```{tip}
You can find much more on Singularity definition files [here](https://sylabs.io/guides/3.7/user-guide/definition_files.html).
```

**Building the container**

We can now build the container with the following command (you need sudo rights for this step):
```console
[me@laptop]$ sudo singularity build example.sif example.def

[... lots of output ...]

INFO:    Adding environment to container
INFO:    Creating SIF file...
INFO:    Build complete: example.sif
```

**Running the container**

Once `example.sif` is generated, we can `scp` the container file to the cluster:

```console
[me@laptop]$ scp example.sif me@saga.sigma2.no
```

First we check the default `g++` compiler on the cluster:
```console
[me@login-1.SAGA ~]$ g++ --version
g++ (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Then we check the `g++` version in the container by running the command through
`singularity exec`:
```console
[me@login-1.SAGA ~]$ singularity exec example.sif g++ --version
g++ (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

We write a simple `hello-world.cpp` program:
```
#include <iostream>

int main() {
    std::cout << "Hello World!" << std::endl;
    return 0;
}
```
and compile it _through_ the container environment:
```console
[me@login-1.SAGA ~]$ singularity exec example.sif g++ hello-world.cpp
[me@login-1.SAGA ~]$ singularity exec example.sif ./a.out
Hello World!
```

Remember that you also need to *run* the program through the container if it is
dynamically linked to some of the containerized libraries.


## Real world example: pdflatex

This example demonstrates:
1. how to build a container from a definition file
2. how to set environment variables inside the container
4. how to document your container
3. how to make your container look like an executable application
4. how to run your container application on the cluster

[Latex](https://www.latex-project.org) is a software package with a plethora of different
package options which can easily mess up your global environment. It is something that is
typically not installed on compute clusters, but could still be useful e.g. for building
code documentation. In this example we will create a fully functional container for the
`pdflatex` command for building PDFs from `tex` files.

**Writing the definition file**

```
Bootstrap: library
From: ubuntu:20.04

%post
    apt-get install -y software-properties-common
    add-apt-repository universe
    apt-get update -y
    apt-get install -y texlive texlive-fonts-extra

%environment
    export LC_ALL=C

%runscript
    pdflatex $@

%labels
    Author Me <me@mymail.com>
    Description PDF latex on a Ubuntu-20.04 base image
    Version v1.0.0

%help
    How to run the container on a tex file:
    $ ./<image-name>.sif <file-name>.tex
```

Here we use the Ubuntu package manager to install a few `texlive` packages on top of a
Ubuntu-20.04 base image, and we set the `LC_ALL` environment variable inside the container
at run time. The `%runscript` section specifies the commands to be run inside the container
when you launch the image file as an executable, where the `$@` will capture an argument string.
In this particular example it means that we can run the image as
```console
$ ./<image-name>.sif <file-name>.tex
```
which will be equivalent of running the given `%runscript` command (`pdflatex` in this case)
through the container with `singularity exec`:
```console
$ singularity exec <image-name>.sif pdflatex <file-name>.tex
```

Finally, we add a few labels (accessible through `singularity inspect <image-name>.sif`) and a help
string (accessible through `singularity run-help <image-name>.sif`) for documentation.

**Building the container**

We build the container on a local computer (requires sudo rights), where we have called the
definition and image files `pdflatex.def` and `pdflatex.sif`, respectively:
```console
[me@laptop]$ sudo singularity build pdflatex.sif pdflatex.def

[... lots of output ...]

    	This may take some time... done.
INFO:    Adding help info
INFO:    Adding labels
INFO:    Adding environment to container
INFO:    Adding runscript
INFO:    Creating SIF file...
INFO:    Build complete: pdflatex.sif
```

**Inpecting the container**

When the image is ready we can inspect the metadata that we put into it

```console
[me@laptop]$ singularity inspect pdflatex.sif
Author: Me <me@mymail.com>
Description: PDF latex on a Ubuntu-20.04 base image
Version: v1.0.0
org.label-schema.build-arch: amd64
org.label-schema.build-date: Thursday_10_June_2021_13:12:27_CEST
org.label-schema.schema-version: 1.0
org.label-schema.usage: /.singularity.d/runscript.help
org.label-schema.usage.singularity.deffile.bootstrap: library
org.label-schema.usage.singularity.deffile.from: ubuntu:20.04
org.label-schema.usage.singularity.deffile.mirrorurl: http://us.archive.ubuntu.com/ubuntu/
org.label-schema.usage.singularity.deffile.osversion: focal
org.label-schema.usage.singularity.runscript.help: /.singularity.d/runscript.help
org.label-schema.usage.singularity.version: 3.7.0
```

```console
[me@laptop]$ singularity run-help pdflatex.sif
    How to run the container on a tex file:
    $ ./<image-name>.sif <file-name>.tex
```

**Running the container**

When we are happy with the container we can move it to any machine where we would like
to run `pdflatex`. Here we `scp` to Saga and log in with `-X` in order to browse the
produced PDF:
```console
[me@laptop]$ scp pdflatex.sif me@saga.sigma2.no
[me@laptop]$ ssh -X me@saga.sigma2.no
```
We write a simple `hello-world.tex` file
```
\documentclass[12pt]{article}
\begin{document}
Hello World!
\end{document}
```
and run our container on it:
```console
[me@login-1.SAGA ~]$ ./pdflatex.sif hello-world.tex
This is pdfTeX, Version 3.14159265-2.6-1.40.20 (TeX Live 2019/Debian) (preloaded format=pdflatex) restricted \write18 enabled.
entering extended mode
(./hello-world.tex
LaTeX2e <2020-02-02> patch level 2
L3 programming layer <2020-02-14>
(/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
Document Class: article 2019/12/20 v1.4l Standard LaTeX document class
(/usr/share/texlive/texmf-dist/tex/latex/base/size12.clo))
(/usr/share/texlive/texmf-dist/tex/latex/l3backend/l3backend-pdfmode.def)
No file hello-world.aux.
[1{/var/lib/texmf/fonts/map/pdftex/updmap/pdftex.map}] (./hello-world.aux) )</u
sr/share/texlive/texmf-dist/fonts/type1/public/amsfonts/cm/cmr12.pfb>
Output written on hello-world.pdf (1 page, 9893 bytes).
Transcript written on hello-world.log.
```

Finally, you can inspect the produced file e.g. in a browser:
```console
[me@login-1.SAGA ~]$ firefox hello-world.pdf
```
where you will hopefully see an almost blank page with the words "Hello World!" written.
