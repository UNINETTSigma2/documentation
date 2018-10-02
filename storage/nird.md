# NIRD - National e-Infrastructure for Research Data

The new data infrastructure, named **NIRD** (National Infrastructure for Research Data),
will provide storage resources with yearly capacity upgrades, data security
through geo-replication (data stored on two physical locations) and adaptable
application services, multiple storage protocol support, migration to third-party
cloud providers and much more. Alongside the national high-performance computing
resources, NIRD forms the backbone of the national e-infrastructure for research
and education in Norway, connecting data and computing resources for efficient
provisioning of services.

The NIRD storage system consists of SFA14K controllers, 10TB NL-SAS drives with
a total capacity of 12PiB in addition to a centralized file system
(IBM GridScaler) supporting multiple file, block and object protocols. Sigma2
will provide the storage infrastructure with resources for the next 4 – 5 years
through multiple upgrades and is expected to triple in capacity during its life-time.

The NIRD infrastructure offers Storage services, Archiving services and  processing capacity for computing on the stored data.  
More info here

[Research Data](https://www.sigma2.no/content/data-storage)


## Project data storage

### Getting Access

To gain access to the storage services, a formal application is needed. The process
is explained at the [User Access](https://www.sigma2.no/node/36) page.

### Logging In

Access to the Project data storage area is through front-end (login) node:

    login.nird.sigma2.no
    
Users must be registered and authorized by the project responsible before obtaining access.

To access or transfer data use the following tools: ssh, scp or stfp. Visit the [Transferring files](https://documentation.sigma2.no/storage/file-transfering.html) page for details.


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
For more information, visit the [Storage Systems on Fram](storagesystems.md) page.

Access to Fram is permitted only trough SSH.
One can use *scp* and *sftp* to upload or download data from Fram.

### File transfering
To transfer files between NIRD and Fram, regular *cp*/*mv* commands can be
used, since NIRD file systems are mounted on Fram.

#### Basic tools (scp, sftp)

* scp - secure copy files between hosts on a network

```
# copy single file to home folder on Fram
# note that folder is ommitted, home folder being default
scp my_file.tar.gz <username>@login.nird.sigma2.no:

# copy a directory to project area
scp -r my_dir/ <username>@login.nird.sigma2.no:/projects/<projectname>/
```

* sftp - interactive secure file transfer program (Secure FTP)

```
# copy all logs named starting with "out" from /projects/project1
# to project1 folder
sftp <username>@login.nird.sigma2.no
sftp> lcd /projects/project1
sftp> cd project1
sftp> put out*.log
```

### Backup

* Geo-replication is set up between Tromsø and Trondheim.
* For backup, snapshots are taken with the following frequency:
    * daily snapshots of the last 7 days
    * weekly snapshots of the last 5 weeks. 
* See [Backup](backup.md).


