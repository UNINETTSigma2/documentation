# NIRD - National Infrastructure for Research Data

**NIRD** is the new National e-Infrastructure for Research Data. It is
owned and operated by UNINETT Sigma2.  It offers services and
capacities to any scientific disciplines that require access to
advanced, large scale or high-end resources for storing, processing,
publishing research data or searching digital databases and
collections.

NIRD will provide storage resources with yearly capacity upgrades,
data security through geo-replication (data stored on two physical
locations) and adaptable application services, multiple storage
protocol support, migration to third-party cloud providers and much
more. Alongside the national high-performance computing resources,
NIRD forms the backbone of the national e-infrastructure for research
and education in Norway, connecting data and computing resources for
efficient provisioning of services.

The NIRD storage system consists of SFA14K controllers, 10 TB NL-SAS
drives with a total capacity of 12 PiB in addition to a centralized
file system (IBM GridScaler) supporting multiple file, block and
object protocols. Sigma2 will provide the storage infrastructure with
resources for the next 4 – 5 years through multiple upgrades and is
expected to triple in capacity during its life-time.

The NIRD infrastructure offers Storage services, Archiving services
and processing capacity for computing on the stored data.  See
the [Research Data](https://www.sigma2.no/content/research-data) page
for an overview of the services.


## Getting Access

To gain access to the storage services, a formal application is needed. The process
is explained at the [User Access](https://www.sigma2.no/node/36) page.


## Logging In

Access to the Project data storage area is through front-end (login) node:

    login.nird.sigma2.no

Note that this hostname is actually a DNS alias for
`login0.nird.sigma2.no`, `login1.nird.sigma2.no`,
`login2.nird.sigma2.no` and `login3.nird.sigma2.no`.  Those are
containers, each one running the image of a login node.  A login
container offers resources for a maximum of 16 cpus and 128GB of
memory.

Users must be registered and authorized by the project responsible
before getting access.

To access or transfer data use the following tools: `ssh`, `scp` or
`sftp`.  Visit the [Transferring files](../faq/file_transfer.md) page
for details.


## Home directories

Each user has a home directory `/nird/home/<username>`, where
`<username>` is the username.  The default quota for home directories
is 20 GiB and 100,000 files.  To check the disk usage and quotas, type

     dusage
     
Home directories on NIRD also contain a backup of Fram home
directories (when relevant) in `/nird/home/<username>/backup/fram`.
To account for this default quota is doubled when relevant.  Note that
this is a _backup_ from Fram; you cannot transfer files to Fram by
putting them here.


## Project area

Each project gets a NIRD project area `/nird/projects/NSxxxxK`,
where `NSxxxxK` is the ID of the project.

The project area has a quota on disk space and the number of files,
and you can see the quota and the current usage by running

    dusage -p NSxxxxK

The NIRD project area is also mounted on the login nodes (but _not_
the compute nodes) of Fram or Saga, when relevant.  For more
information, visit the [Fram and Saga](clusters.md) page.
(Note that the mounts on Saga do not exist yet, but will be there
shortly.)
