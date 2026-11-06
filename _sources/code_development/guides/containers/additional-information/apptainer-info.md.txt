(apptainer-info)=
# Filesystem Mounts in Apptainer
Apptainer provides seamless host integration out of the box. If you 
run `whoami` or `ls` inside a newly pulled public container, you will 
notice that it can already see your local files and perfectly recognizes 
your username.

This is achieved through Dynamic User Mapping. When the container boots, 
it automatically reads your current user details from the host machine's 
`/etc/passwd` and `/etc/group` files and injects them directly into the 
container. As a result, you remain exactly who you are inside the 
environment, maintaining your identical host system permissions 
without needing root access.

## **Automatic Bind Mounts** 

While the default system paths are mounted automatically, understanding 
how binding works is crucial for managing your storage. Rather than 
duplicating massive files, binding directly links the host cluster's 
filesystem paths into the container. This allows the container to 
transparently read and write to shared HPC storage in real time.

Apptainer automatically maps key directories from the host 
cluster directly into the container´s file structure. By default, 
your home directory `$HOME`, your current directory `$PWD`, and `/tmp` 
are made visible inside the container. Since this is not the copy, 
any files modified or saved while inside the container are written 
diretly to the host´s physical storage and will persist after the 
container shuts down.

However, if you want to access shared parallel 
cluster filesystems like `/scratch` or `/cluster/projects` you need to 
pass an explicit flag at runtime to make them visible. 

```bash
singularity shell --bind /scratch my-container.sif
``` 

Refer to the section below to see how it works.

## Example of accessing the project area from the container

Lets say if you create a directory named `data` inside the project area 
where you have this container image `hello-world.sif `. Inside the data 
directory you create a file `input.txt` that will print from digit 1 to 4 
when you run it. 

Now , if you attempt to read data from a project directory without explicit 
access, the container will throw a `No such file or directory` error.

```console
$ apptainer exec hello-world.sif head -n2 /cluster/projects/nnxxxxk/containers/data/input.txt
/usr/bin/head: cannot open '...': No such file or directory
```

Now, to grant the container access to cluster storage, you must use 
the `--bind` (or -B) flag. This maps a folder on the cluster host machine 
to a folder inside the running container.


```console
$ apptainer exec --bind /cluster/projects/nnxxxxk hello-world.sif head -n2 /cluster/projects/nnxxxxk/containers/data/input.txt
```
 This way, keeps the file path identical both inside and outside the container
 and you will be able to print those numbers.

 ## Example of Apptainer run and Apptainer exec command

 Firstly, lets inspect what is inside this `hello-world.sif` container using 
 this command.

 ```console
$ apptainer inspect -r hello-world.sif
#!/bin/sh

exec /bin/bash /rawr.sh
```

 This means that the `hello-world.sif` image has a user defined command 
 called `rawr.sh`. You could manually open this file using this command which 
 will give the same output. 

```bash
 apptainer exec hello-world.sif cat /singularity
```

 Now using the `exec` command you can print the contents of that script 
 using as shown here.

 ```console
$ apptainer exec hello-world.sif cat /rawr.sh
#!/bin/bash

echo "RaawwWWWWWRRRR!! Avocado!"
```

 In order to make the container run the script you can use the `run ` 
 command as shown like this:

 ```console
$ apptainer run hello-world.sif
RaawwWWWWWRRRR!! Avocado!
```

 However, if you try to use the `run` command and try to print the 
 os-release of that container, it still will print the content of 
 the `rawr.sh` as shown here:

 ```console
$ apptainer run hello-world.sif cat /etc/os-release
RaawwWWWWWRRRR!! Avocado!
```

 Hence, in this case you should use `exec` command.
 ```console
$ apptainer exec hello-world.sif cat /etc/os-release
NAME="Ubuntu"
VERSION="14.04.6 LTS, Trusty Tahr"
```

