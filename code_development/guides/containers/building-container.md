(building-container)=
# Building the Container

Building a Apptainer container creates an unchaning, single-file snapshot 
of your software environment that can be tracked via version control alongside 
your research.This ensures you can instantly rebuild or share the exact same 
runtime setup on any machine, guaranteeing that your secientific results remain 
completely reproducible years down the line. 

There are many ways to build a Apptainer images. You can read more about them here 
[Build a container](https://apptainer.org/docs/user/main/build_a_container.html).
However, we will discuss about one of the most commonly used approach which is 
building from a  
[Apptainer Definition file](https://apptainer.org/docs/user/main/build_a_container.html#building-containers-from-apptainer-definition-files)

## Apptainer Definition File
A Apptainer definition file is a text file that specifies the instructions and 
configuration needed to build a container image.
Below is the simple illustration of definition file which only use the `%post` 
and `%runscript` sections. More details can be found here 
[Definition File Official Documentation](https://apptainer.org/docs/user/main/definition_files.html)

```bash
Bootstrap: docker
From: ubuntu:20.04

%post
  apt-get -y update && apt-get install -y python3

%runscript
  python3 -c 'print("Hello World!")'
```
The `%` prefix is used to define different configurations during the build process. 
In the above example, we specify the backend protocol to fetch the base image as 
docker. This tells Apptainer to pull the layers from a Docker registry , automatically 
handles the download and convert it into Singularity´s format on the fly. You cannot 
build a container using a competely blank file. Any software or code you write requires 
basic system utilities, core C libraries, and folder structures to interact with the 
system kernel. The most efficient approach is to start with an existing Base Image 
that already provides this minimal operating system user space. This is the reason 
why we pull a clean, minimal deployment of `Ubuntu 20.04`.

The `%post` block is where you actually install your applications, compile code and 
configure your container´s environment. The command written in the `%post` section 
is an automated installation script. Every command here is executed as standard shell 
code inside the base operating system you defined in the header. So, in this case 
here this command will run within your minimal Ubuntu 20.04 layout to update the 
package manager index and install `Python 3`. We must bypass any interactive confirmation 
prompts `Do you want to continue? [Y/n]` by explicitly forcing a "yes" response which 
is the reason why we passed `-y` flags. Beyond basic package managers, you can use
the `%post` section to run `pip install` commands, download remote datasets via `wget` 
or `curl`, create directories and clone repositories. Everything modified here is 
permanently baked into the final, read-only `.sif` image. 

Finally, we have the `%runscript` section which is used to define a script that we want to run using the 
`apptainer run ` command when a container is started. However, this part is optional in the definition file.

Once you save this definition file you can build a `.sif` image using this command.

`apptainer build my_container.sif <definition_file>.def`

```{note}
You cannot build it on the login node, so you have to build it on the compute node. Also, depending on the
implementation on each of our cluster, you might need to pass `--fakeroot` flag in the build command above if it 
fails.

```

```{toctree}
:maxdepth: 1
:hidden:

```
