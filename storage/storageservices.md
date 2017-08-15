# Storage Services

The table below contains a limited overview of the current storage hardware resources.

| Resource | 	Description | 	Specification | 	Media type | 	Total disk capacity | 	Service interface |
| :------------- | :------------- | :------------- | :------------- | :------------- | :------------- |
| norstore-osl | 	Serves project area and data archive | 	Hitachi Unified Storage System | 	HNAS disk | 	3.1 PiB | 	command line and integrated |
| norstore-osl | 	Serves services for Sensitive data | 	Hitachi Unified Storage System | 	HNAS disk | 	0.48 PiB | 	two factor authentication, command line and integrated |
| norstore-osl-tape | 	Serves project area and data archive | 	Spectra Logic T-finity | 	TS1140 tape | 	4.0 PiB | 	command line and integrated |
| norstore-tos | 	Serves project area and data archive | 	Hitachi Unified Storage System | 	HNAS disk | 	0.5 PiB | 	command line and integrated |

For more information about the system overview of the resources, visit the [Astrastore - teknisk systembeskrivelse](http://www.uio.no/tjenester/it/hosting/storage/mer-om/astrastore-teknisk.html) page.

# Getting Access

To gain access to the storage services, a formal application is needed. The process
is explained at the [User Access](https://www.sigma2.no/node/36) page.

# Logging In

Access to the Project data storage area is through front-end (login) nodes. Users must be registered and authorized by the project responsible before obtaining access.

| Resource |	Server URI (access point) |	Purpose |
| :------------- | :------------- | :------------- |
| norstore-osl front-end | 	login.norstore.uio.no | 	general |
| norstore-osl (high perf) | 	login3.norstore.uio.no | 	general, not hpn-ssh enabled |

To access or transfer data to one of these resources, use the following tools: ssh, scp or stfp. Visit the [Transferring files](storage/file-transfering.md) page for details.

# Storing Files on Fram

The [Storage Systems on Fram](storagesystems.md) page has the list of places to store files when running jobs on Fram.
