# NIRD Service Platform

The NIRD Service Platform (NIRD SP) is a Kubernetes-based cloud platform 
providing persistent services such as web services, domain- and community specific portals, 
as well as on-demand services through the [NIRD Toolkit](/nird_toolkit/overview_nird_toolkit.md). 

The kubernetes solution of the NIRD SP enables researchers to run microservices for pre/post-processing, data discovery and analysis as well as data sharing, regardless of dataset sizes stored on NIRD.
To start using the NIRD SP you need to follow the same procedure as enabling [NIRD Toolkit](/nird_toolkit/overview_nird_toolkit.md) 

## NIRD SP Hardware information
The technical specifications of the NIRD SP cluster are listed below:



|NIRD Service platform                       | Total           | Details                                                                 |
|:----------------------|:----------------|-------------------------------------------------------------------------|
| Number of worker nodes | 	12             |                                                                         |
| CPUs                  | 	2368 cores     | 8 workers with 256 cores, 4 workers with 80 cores                       |
| GPUs                  | 	30 Nvidia V100 |                                                                         |
| RAM	                | 	9 TiB          | 4 workers with 512 GiB, 4 workers with 1024 GiB, 4 workers with 768 GiB |
| Interconnect          | 	 Ethernet      | 8 workers with 2 x 100 Gbit/s, 4 workers with 2 x 10 Gbit/s             | 

## Types of access to interact with NIRD SP

### Use on-demand data analytic services through NIRD Toolkit. 
Check out the user guide for the 
[NIRD Toolkit](/nird_toolkit/overview_nird_toolkit.md)


### Run you favourite web-based services
On the NIRD Service Platform, you can run your favourite web-based tools or portals to analyse and visualise massive amounts of data stored on NIRD – all in one place, without moving data.
You can also use the NIRD Service Platform to support S3 services.
In addition, we can provide an apache web service or an OPeNDAP Hyrax server to share data. 
To launch a web-based service / portal with a permanent web address, send a request to contact@sigma2.no.

### Dedicated virtual environment
On the NIRD Service Platform, you can get a dedicated virtual environment (we call it a “login container”) to run data-intensive pre-/post-processing analysis without queuing the system and without moving data. 
The login containers can be customised with tools specific to the project and powered with virtual CPUs, RAM and eventually GPUs.
To get a dedicated virtual environment send a request to contact@sigma2.no

### Kubernetes dashboard access
In the kubernetes dashboard accessible at https://db.nird.sigma2.no, you can get an overview of running services connected to your NIRD project space. Here you can also find logs from the services that has been set up using NIRD Toolkit. 

### Kubectl access
Contact the NRIS support (support@nris.no) in case you need CLI access to the NIRD SP. This is mostly relevant for research groups that are already familiar with kubernetes.  