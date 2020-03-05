# The Norwegian Academic HPC Services


## Introduction

The current Norwegian academic HPC infrastructure is maintained by an organization called the [Metacenter](https://sigma2.no/metacenter), which is a joint collaboration between the four oldest Universities in Norway ([UiO](https://uio.no), [UiB](https://uib.no), [NTNU](https://ntnu.no) and [UiT](https://uit.no)) and [Sigma2](https://www.sigma2.no/).

This documentation aims to be user-friendly, readily available and correct. The future will be our judges in how well we succeed in this.

<u>We also hold a set of ambitions for these pages:</u>

1. They should be used and read.
2. Users (<b>you</b>) should be able to contribute through pull requests.
3. They should be relevant for us, you users and for external colleagues.


### Acknowledging use of national HPC infrastructure

Projects are required to acknowledge the use of the national e-infrastructure resources in their scientific publications. Papers, presentations and other publications that feature work that relied on Sigma2 should include such an acknowledgement. 

UNINETT Sigma2 has defined the following template for such acknowledgements:

	"The computations/simulations/[SIMILAR] were performed on resources provided by 
	UNINETT Sigma2 - the National Infrastructure for High Performance Computing and
	 Data Storage in Norway"

--------------------------------
## For the beginner
If you are new here, you might want to learn the basics first here:

* [Getting started](quick/getttingstarted.md)
* [Latest changes and events](https://opslog.sigma2.no)
* [Editing files](faq/emacs.md)
* [Password-less login](faq/ssh.md)

### Training
* [Training calendar](https://www.sigma2.no/events)
* [HPC carpentry](support/hpc_carpentry.md)
* [Introduction to HPC training material](https://sabryr.github.io/hpc-intro/)


## Getting help and access
* [Support line](help/support.md)
* [Writing good support requests](help/how_to_write_good_support_requests.md)
* [Lost or expiring password](help/password.md)
* [Applying for user accounts](help/account.md)
* [Applying for resources](help/resources.md)

## Current status and announcements
* [Hardware live status](https://www.sigma2.no/hardware-status)
* [Latest changes and events](https://opslog.sigma2.no)

--------------------------------
## For advanced users
#### Jobs
* [Queue System](jobs/queue_system.md)
* [Job Scripts](jobs/job_scripts.md)
* [Managing Jobs](jobs/managing_jobs.md)
* [Interactive Jobs](jobs/interactive_jobs.md)
* [Projects and accounting](jobs/projects.md)
* [Guides](jobs/guides.md)

#### Software
* [Software Module Scheme](apps/modulescheme.md)
* [Installing software as user](apps/userinstallsw.md)
* [Installed Software](apps/which_software_is_installed.md)
* [Application guides](apps/appguides.md)


#### Storage and file managment
Fram and Saga use the NIRD storage system for storing archives for other research data. NOTUR projects have access
to this geo-replicated storage through various methods.

* [NIRD](storage/nird.md)
* [Fram and Saga](storage/clusters.md)
* [Backup](storage/backup.md)
* [Sharing files](storage/data_policy.md)
* [Transferring files](storage/file_transfer.md)
* [Performance tips](storage/performance/overview.md)
    * [Lustre (Fram and Stallo)](storage/performance/lustre.md)
    * [BeeGFS (Saga)](storage/performance/beegfs.md)
    * [What to avoid](storage/performance/what_to_avoid.md)

#### Code developtment
* [Compilers](development/compilers.md)
* [Debugging](development/debugging.md)
* [Performance Analysis and Tuning](development/performance.md)


## About UNINETT Sigma2
UNINETT Sigma2 AS manages the national infrastructure for computational science in Norway, and offers services in high performance computing and data storage.
Visit https://www.sigma2.no for more information.

Latest news and announcements from Metacenter are posted at <a href="https://opslog.sigma2.no" target="_blank">Metacenter OpsLog</a> and the @MetacenterOps Twitter channel.
