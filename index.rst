======================================================================
The Norwegian academic high-performance computing and storage services
======================================================================

The Norwegian academic high-performance computing and storage
infrastructure is maintained by the `Sigma2
Metacenter <https://sigma2.no/metacenter>`__, which is a joint
collaboration between `UiO <https://www.uio.no>`__,
`UiB <https://www.uib.no>`__, `NTNU <https://www.ntnu.no>`__,
`UiT <https://uit.no>`__, and `UNINETT
Sigma2 <https://www.sigma2.no/>`__.

This website (https://documentation.sigma2.no/) holds technical
documentation about the compute and storage resources. For more general
information and service overview, please also see https://www.sigma2.no.

**Latest news and announcements** are posted at the `Metacenter
OpsLog <https://opslog.sigma2.no>`__ and the
`@MetacenterOps <https://twitter.com/MetacenterOps>`__ Twitter channel.

--------------

Compute, storage, pre/post-processing, visualization, machine learning
----------------------------------------------------------------------

We offer compute resources (`Betzy <hpc_machines/betzy.html>`__,
`Fram <hpc_machines/fram.html>`__, and `Saga <hpc_machines/saga.html>`__, storage resources
(`NIRD <files_storage/nird.html>`__), as well as the `NIRD
Toolkit <https://www.sigma2.no/nird-toolkit>`__ platform for pre- and
post-processing analysis, data intensive processing, visualization,
artificial intelligence, and machine learning.


First time on a supercomputer?
------------------------------

Please read the **GETTING STARTED** section (left sidebar). In the
sidebar overview you will also find technical details about the
machines, instructions for using installed software, for submitting
jobs, storage, and code development.

Please do not hesitate to write to support@metacenter.no if you find
documentation sections which are not clear enough or have suggestions
for improvements. Such a feedback is very important to us and will
count.


How to get the most out of your allocation
------------------------------------------

We want to support researchers in getting the most out of the
high-performance computing services. When supporting users, we see that
these problems are very frequent:

- **Reusing outdated scripts** from colleagues without adapting them to
  optimal parameters for the cluster at hand and thus leaving few cores
  idle. Please check at least how many cores there are on a particular
  cluster node.
- **Requesting too much memory** which leads to longer queuing and less
  resource usage. Please check `how to choose memory
  settings <jobs/choosing_memory_settings.html>`__.
- **Requesting more cores than the application can effectively use** without
  studying the scaling of the application. You will get charged more than
  needed and others cannot run jobs. If others do this, your own jobs queue.
- **Submitting jobs to the wrong queue** and then queuing longer than
  needed. Please take some time to study the different `job types
  <jobs/choosing_job_types.html>`__.

If you are unsure about these, please contact us via
support@metacenter.no and we will help you to use your allocated
resources more efficiently so that you get your research results faster.

--------------

Acknowledging use of national HPC infrastructure
------------------------------------------------

Projects are required to acknowledge the use of the national
e-infrastructure resources in their scientific publications. Papers,
presentations and other publications that feature work that relied on
Sigma2 should include an acknowledgement following this template:

::

   The computations/simulations/[SIMILAR] were performed on resources provided by
   UNINETT Sigma2 - the National Infrastructure for High Performance Computing and
   Data Storage in Norway


Text is licensed CC-BY
----------------------

Unless explicitly noted, all text on this website is made available
under the `Creative Commons Attribution license
(CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`__ with
attribution to the Sigma2/Metacenter.

--------------

Index of keywords
-----------------

:ref:`genindex`

--------------

.. toctree::
   :maxdepth: 1
   :caption: News

   Latest changes and events <https://opslog.sigma2.no>
   Hardware live status <https://www.sigma2.no/hardware-status>


.. toctree::
   :maxdepth: 1
   :caption: Getting help

   getting_help/support_line.md
   getting_help/how_to_write_good_support_requests.md
   getting_help/qa-sessions.md
   getting_help/lost_forgotten_password.md
   getting_help/project_leader_support.md
   getting_help/advanced_user_support.md


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/applying_account.md
   getting_started/applying_resources.md
   getting_started/training.md
   getting_started/getting_started.md
   getting_started/editing_files.md
   getting_started/create_ssh_keys.md


.. toctree::
   :maxdepth: 1
   :caption: Services

   nird_archive/user-guide.md
   nird_toolkit/overview.rst
   services/easydmp-user-documentation.md
   getting_help/course_resources.md


.. toctree::
   :maxdepth: 1
   :caption: High-performance computing

   hpc_machines/hardware_overview.md
   hpc_machines/betzy.md
   hpc_machines/fram.md
   hpc_machines/saga.md
   hpc_machines/migration2metacenter.md
   jobs/overview.rst
   code_development/overview.rst
   computing/tuning-applications.md


.. toctree::
   :maxdepth: 1
   :caption: Software

   software/modulescheme.md
   software/installed_software.md
   software/userinstallsw.rst
   software/containers.md
   software/appguides.md
   software/licenses.md


.. toctree::
   :maxdepth: 1
   :caption: Files and Storage

   files_storage/nird.md
   files_storage/clusters.md
   files_storage/backup.md
   files_storage/sharing_files.md
   files_storage/file_transfer.md
   files_storage/performance.md
