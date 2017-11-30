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

There are two main uses for NIRD and visit the following pages for more information.
1. [Research Data](https://www.sigma2.no/content/data-storage)
2. [Storage of scientific project data](https://www.sigma2.no/content/storage-scientific-project-data)

Fram has a NIRD directory `/nird` that projects can use. For more information, visit the [Storage Systems on Fram](storagesystems.md) page.

## Project data storage

### Getting Access

To gain access to the storage services, a formal application is needed. The process
is explained at the [User Access](https://www.sigma2.no/node/36) page.

### Logging In

Access to the Project data storage area is through front-end (login) node:

    login.nird.sigma2.no
    
Users must be registered and authorized by the project responsible before obtaining access.

To access or transfer data use the following tools: ssh, scp or stfp. Visit the [Transferring files](storage/file-transfering.md) page for details.

### Project area

Project areas are located in `/nird/projects/<project_ID>`.

The project area is quota controlled and current usage is obtained by running the command:

    dusage -p <project_ID>


**Notes:**
* Geo-replication is set up between Tromsø and Trondheim.
* For backup, snapshots are taken with the following frequency:
    * daily snapshots of the last 7 days
    * weekly snapshots of the last 5 weeks. 
* See [Backup](backup.md).


