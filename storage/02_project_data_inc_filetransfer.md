## Project data storage

### Getting Access

To gain access to the storage services, a formal application is needed. The process
is explained at the [User Access][ualink] page.

### Logging In

Access to the Project data storage area is through front-end (login) node:

    login.nird.sigma2.no

Note that this hostname is actually a DNS alias for:   
login1.nird.sigma2.no, login2.nird.sigma2.no, login3.nird.sigma2.no, login4.nird.sigma2.no   
those are containers each one running the image of a login node.   
A login container offers resources for a maximum of 16 cpus and 128MB of memory.

Users must be registered and authorized by the project responsible before obtaining access.

To access or transfer data use the following tools: ssh, scp or stfp. Visit the [Transferring files][tflink] page for details.


### Home directories

Home directories are located in `/nird/home/<username>`.
Quota for home is 20GB and 100000 files. To check the disk usage type

     dusage
     
Home directories are also visible at /nird/home from FRAM login nodes.   
Since those are mounted with NFS on FRAM it is very important not to use home directories   
as storage for jobs running of FRAM.

### Project area

NIRD project areas are located in `/nird/projects/<project_ID>`.

The project area is quota controlled and current usage is obtained by running the command:

    dusage -p <project_ID>

FRAM projects are only available from FRAM login nodes.   
For more information, visit the [Storage Systems on Fram][sslink] page.



### File transfering
Access to NIRD is permitted only through SSH.
One can use *scp* and *sftp* to upload or download data from NIRD.

* scp - secure copy files between hosts on a network

```
# copy single file to home folder on NIRD
# note that folder is ommitted, home folder being default
scp my_file.tar.gz <username>@login.nird.sigma2.no:

# copy a directory to project area
scp -r my_dir/ <username>@login.nird.sigma2.no:/projects/<projectname>/
```

* sftp - interactive secure file transfer program (Secure FTP)

```
# copy all logs named starting with "out" from project1 folder
# to /projects/project1
sftp <username>@login.nird.sigma2.no
sftp> cd /projects/project1
sftp> lcd project1
sftp> put out*.log
```

