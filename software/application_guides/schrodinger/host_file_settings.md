---
orphan: true
---

# Keywords for schrodinger.hosts file settings

| Keyword      | Description           |
| ------------- |:-------------:| 
|base |Name of an entry (the base entry) that is the basis for the current entry. All the keywords from the base entryare inherited by the current entry, and new keywords may be added, in any order. A base entry can include another base entry.|
|env | Environment variables to be set on the host. The syntax for the environment variables is variable=value, regardless of the shell used. List each environment variable on a separate env line.|
|gpgpu | Specify a graphics processor (GPU) to use on the host. One instance should be used for each GPU specified. The specification is in the form id, description, where id is the numerical GPU id, usually starting from 0, and description is the description of the GPU, for example Tesla V100.|
|host |Host name. This entry is only needed if it is different from the name setting or if the queueing software is only available on a particular host. Not valid in the localhost entry.|
|serverhost	|Name of host used to stage job output when the host from which the job was submitted is offline. This might be the head node of a cluster, for example. This setting is ignored if the job submission host does not have offline job management enabled.|
|include | Name of an auxiliary hosts file to be included in the current hosts file. The inclusion is done by replacing the include line with the contents of the specified file.|
|knime |Path to an external KNIME installation (i.e. an installation other than the one in the Schrödinger installation).|
|name| Name of the host entry or batch queue. For a host this is usually the host name. This name is displayed in the Start dialog box. The name must not contain spaces. The value localhost is a special name that means the host on which the job is launched.|
|nodelist| List of entry names, used to define a multiple-host entry. A name may be followed by a colon and a number of processors. Can be combined with a host setting.|
|parallel| Specify whether the host supports MPI parallel jobs or not. The value can be specified as yes or no, true or false, 1 or 0.|
|port| Server port to use when sending jobs to a server (Used by KNIME only).|
|processors| Number of processors available on the host. If the host is part of a cluster, this number should be the total number of processors available on the cluster. For multicore processors, the number should be the total number of cores available. The default is 1, except for the localhost entry, where the default is the number of available processors (or cores).|
|processors_per_node| Number of processors (cores) per node available to a batch queue. This setting is used by applications that support threaded parallel execution (OpenMP).|
|proxyhost| Host on which to run jproxy. This setting should be made when the host from which a job is launched cannot open a socket connection to the host on which the job is actually run. By default, jproxy is run on the host specified by the host keyword, and is only run when using a queuing system. This setting is only needed in cases where using the default is impossible or impractical. Only valid when the host entry also contains a queue setting.|
|proxyport| Specify the port or range of ports that jproxy may use. Ports can be specified as comma or colon-separated lists without spaces. Ranges can specified with a dash, for example, 5987:5989-5992:5994. Only valid when the host entry also contains a queue setting.|
|qargs| Arguments to be used when submitting jobs to a batch queue. These arguments should specify any parameters that define the queue.|
|queue| Queuing system name, which is the subdirectory of $SCHRODINGER/queues that contains the support files for the queuing system. PBS10.4, SGE, LSF, Torque, and Slurm are the supported systems. Not valid in the localhost entry.|
|recoverjobs| Disable recovery of failed jobs if set to no. Use this setting only for jobs where job recovery might not be possible (such as on the cloud).|
|schrodinger| The path to the Schrödinger software installation on the host. Not valid in the localhost entry.|
|tmpdir| Base directory for temporary or scratch files, also called the scratch directory. The file system on which this directory is mounted should be large enough for the largest temporary files, should be mounted locally, and should be writable by the user.|
|user| User name to use on the host. This should never be set in the hosts file in the installation directory. It is required if the user has a different user name on the defined host than on the host on which the job is launched.|

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)