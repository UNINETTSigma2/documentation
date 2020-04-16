# NIRD - National Infrastructure for Research Data

**NIRD** is the **N**ational e-**I**nfrastructure for **R**esearch **D**ata. It
 is owned and operated by [UNINETT Sigma2](https://www.sigma2.no).

<div class="alert alert-info">
  <p>
    The NIRD infrastructure offers storage services, archiving services, and
    processing capacity for computing on the stored data.  It offers services
    and capacities to any scientific discipline that requires access to
    advanced, large scale, or high-end resources for storing, processing,
    publishing research data or searching digital databases and collections.
  </p>
</div>

NIRD will provide storage resources with yearly capacity upgrades,
data security through geo-replication (data stored on two physical
locations) and adaptable application services, multiple storage
protocol support, migration to third-party cloud providers and much
more. Alongside the national high-performance computing resources,
NIRD forms the backbone of the national e-infrastructure for research
and education in Norway, connecting data and computing resources for
efficient provisioning of services.


## Technical specifications

The NIRD storage system consists of DDN SFA14K controllers, 3400 x 10TB NL-SAS
drives with a total capacity of 2 x 11 PiB.
The solution is based on DDN GridScaler® parallel file system, supporting
multiple file, block and object protocols.


## Getting access

To gain access to the storage services, a formal application is required. The
process is explained at the
[How to apply for a user account](https://www.sigma2.no/how-apply-user-account)
page.

Users must be registered and authorised by the project responsible
before getting access.

To access or transfer data, you may use the following tools: `ssh`, `scp` or
`sftp`.  Visit the [transferring files](file_transfer.md) page
for details.


## Logging in

Access to your $HOME on NIRD and the project data storage area is through the
login containers.
Login containers are running on servers directly connected to
the storage on both sites -that is Tromsø and Trondheim- to facilitate data
handling right where the primary data resides. Each login container offers a
maximum of 16 CPU cores and 128GiB of memory.

Login containers can be accessed via following addresses:
```
login-tos.nird.sigma2.no
login-trd.nird.sigma2.no
```

<div class="alert alert-info">
  <p>
    Note that we run four login containers per site. <br />
    If you plan to start a `screen` session on one of the login containers or
    you wish to copy data with the help of `scp` or `WinSCP`, you should log in
    to a specific container.
  </p>
  <ul>Addresses are:
    <li>login<strong>X</strong>-tos.nird.sigma2.no</li>
    <li>login<strong>X</strong>-trd.nird.sigma2.no</li>
    <li><strong>X</strong> - can have values between 0 and 3.
  </ul>
</div>


## Home directories

Each user has a home directory `/nird/home/<username>`, where
`<username>` is the username.  The default quota for home directories
is 20 GiB and 100 000 files.  To check the disk usage and quotas, type:
```
$ dusage
```

Home directories on NIRD also contain a backup of Betzy, Fram and Saga home
directories (when relevant) in `/nird/home/<username>/backup/fram` and
`/nird/home/<username>/backup/saga`.
To account for this default quota is doubled when relevant.
Note that this is a _backup_ from the HPC cluster; you cannot transfer
files to the cluster by putting them here.


## Project area

Each NIRD Storage project gets a project area `/nird/projects/NSxxxxK`,
where `NSxxxxK` is the ID of the project.

### Project locality

NIRD Storage projects are -with some exceptions, mutually agreed with the
project leader- stored on two sites and asynchronously geo-replicated.

For every project storage there is a primary data volume on one site and a full
replica on the other site. The main purpose for the replica is to ensure
security and resilience in case of large damage at the primary site. The primary
 site is chosen for operational convenience to be the one closest to where the
 data is consumed, namely NIRD-TOS, if data is analysed on Fram or NIRD-TRD if
 data is analysed on Saga or Betzy HPC clusters.

 Projects have the possibility to read from and write to the primary site, while
 they cannot read from or write to the replica site.

<div class="alert alert-warning">
  <p>
    The users should log onto the login container nearest to the primary data
storage.
  </p>
</div>


### Disk usage

The project area has a quota on disk space and the number of files,
and you can see the quota and the current usage by running:
```
$ dusage -p NSxxxxK
```

### Snapshots

In addition to geo-replication NIRD supports snapshots of project areas
and home directories allowing for recovery of deleted data.
For more information, visit the [backup](backup.md) page.


### Mounts on HPC

When relevant, the NIRD Storage project areas are also mounted on the login
nodes of Betzy, Fram or Saga HPC clusters.

<div class="alert alert-info">
  <p>
    Only the primary data volumes for projects are mounted to the HPC clusters.
  </p>
  <ul>i.e.
    <li>projects from NIRD-TOS to Fram</li>
    <li>projects from NIRD-TRD to Betzy and Saga</li>
  </ul>
</div>


<div class="alert alert-warning">
  <p>
    To avoid performance impact and operational issues, NIRD $HOME and project
    areas are _not_ mounted on none of the compute nodes of the HPC clusters.
  </p>
</div>

For more information, visit the [Betzy, Fram and Saga](clusters.md) page.
