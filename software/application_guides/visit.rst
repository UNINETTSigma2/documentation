:orphan:


VisIt visualization
===================

VisIt visualization is an open-source interactive parallel visualization tool
for viewing scientific data. VisIt can be used for visualizing large 2D and 3D
datasets on both structured and unstructured grids. To find out more, visit the
`VisIt website <https://visit-dav.github.io/visit-website/>`_.


Loading VisIt
-------------

To see which versions of VisIt are available, use

.. code-block:: bash

   > module spider VisIt

Currently, VisIt is installed on Betzy (VisIt v3.1.4) and Fram (VisIt v2.13.0).
To use VisIt, load the desired version by

.. code-block:: bash

   > module load VisIt/<version>

where ``version`` is replaced by the corresponding VisIt version. 

Using VisIt
-----------

VisIt can be used interactively directly on the Sigma2 machines or as a client-server interactive process where the compute engines run on the server and only the image is rendered on the client screen.
In addition, VisIt can take a Python script as an input file and be used as a regular piece of software passed in through the batch system.
See the `VisIt Python documentation <https://visit-sphinx-github-user-manual.readthedocs.io/en/develop/cli_manual/index.html>`_ for details.

Interactive visualization
_________________________

To launch VisIt interactively, one must first request an interactive job (see :ref:`interactive-jobs`) and forward the graphics over SSH (i.e. log into the machine with ``ssh -X username@machine``). 

Next, launch VisIt (possibly in parallel) by

.. code-block:: bash

   > visit -np <number>

where ``-np <number>`` specifies the number of processors to use.
Omitting the ``-np <number>`` argument will launch VisIt in serial. 


Client-server visualization
___________________________

VisIt can be used as client-server visualization tool by having a server render the visualization on the client's screen.
In this case the user will open VisIt locally on a workstation but open and render files on the server.
Below, we provide a quickstart guide for setting up client-server visualization, as well as step-by-step instructions for setting up new host and launch profiles from scratch. 

.. warning::

   To use client-server visualization, the *major* version of VisIt must be the same on both the client and server side.
   E.g. if the version installed on the server is ``3.1.x`` the client's version *must* be ``3.1.y``.

   Note that the local (client) does *not* need to be built for parallel visualization.

Quickstart using existing hosts and launch profiles
***************************************************

For quickly getting started with client-server visualization, existing host profiles with a template launch profile for both Fram and Betzy are provided below.
To use these profiles the user must copy the contexts of the XML files below to the user's VisIt configuration.
This is *usually* located in ``~/.visit/hosts/``, and the files can be stored as e.g. :file:`~/.visit/hosts/host_betzy_sigma2_no.xml`.
The username and account number in the XML files below are only placeholders and must be replaced, either directly in the XML file or through the host and launch configuration in VisIt (see :ref:`visit_host_config`).

If the host profile is configured correctly, the user will see additional available hosts when opening files through VisIt. 

.. code-block:: xml
   :caption: Host and launch profile template for Betzy.
	     
   <?xml version="1.0"?>
   <Object name="MachineProfile">
    <Field name="hostNickname" type="string">betzy.sigma2.no</Field>
    <Field name="host" type="string">betzy.sigma2.no</Field>
    <Field name="userName" type="string">username</Field>
    <Field name="hostAliases" type="string"></Field>
    <Field name="directory" type="string">/cluster/software/VisIt/3.1.4-linux-x86_64-rhel7-wmesa</Field>
    <Field name="shareOneBatchJob" type="bool">false</Field>
    <Field name="sshPortSpecified" type="bool">false</Field>
    <Field name="sshPort" type="int">0</Field>
    <Field name="sshCommandSpecified" type="bool">false</Field>
    <Field name="sshCommand" type="stringVector">"ssh" "-X" </Field>
    <Field name="useGateway" type="bool">false</Field>
    <Field name="gatewayHost" type="string"></Field>
    <Field name="clientHostDetermination" type="string">MachineName</Field>
    <Field name="manualClientHostName" type="string"></Field>
    <Field name="tunnelSSH" type="bool">true</Field>
    <Field name="maximumNodesValid" type="bool">false</Field>
    <Field name="maximumNodes" type="int">1</Field>
    <Field name="maximumProcessorsValid" type="bool">false</Field>
    <Field name="maximumProcessors" type="int">1</Field>
    <Object name="LaunchProfile">
        <Field name="timeout" type="int">480</Field>
        <Field name="numProcessors" type="int">128</Field>
        <Field name="numNodesSet" type="bool">true</Field>
        <Field name="numNodes" type="int">1</Field>
        <Field name="partitionSet" type="bool">false</Field>
        <Field name="partition" type="string"></Field>
        <Field name="bankSet" type="bool">true</Field>
        <Field name="bank" type="string">nnXXXXk</Field>
        <Field name="timeLimitSet" type="bool">true</Field>
        <Field name="timeLimit" type="string">00:30:00</Field>
        <Field name="launchMethodSet" type="bool">true</Field>
        <Field name="launchMethod" type="string">srun</Field>
        <Field name="forceStatic" type="bool">true</Field>
        <Field name="forceDynamic" type="bool">false</Field>
        <Field name="active" type="bool">false</Field>
        <Field name="arguments" type="stringVector"></Field>
        <Field name="parallel" type="bool">true</Field>
        <Field name="launchArgsSet" type="bool">true</Field>
        <Field name="launchArgs" type="string">"--account=nnXXXXk --qos=preproc"</Field>
        <Field name="sublaunchArgsSet" type="bool">false</Field>
        <Field name="sublaunchArgs" type="string"></Field>
        <Field name="sublaunchPreCmdSet" type="bool">false</Field>
        <Field name="sublaunchPreCmd" type="string"></Field>
        <Field name="sublaunchPostCmdSet" type="bool">false</Field>
        <Field name="sublaunchPostCmd" type="string"></Field>
        <Field name="machinefileSet" type="bool">false</Field>
        <Field name="machinefile" type="string"></Field>
        <Field name="visitSetsUpEnv" type="bool">false</Field>
        <Field name="canDoHWAccel" type="bool">false</Field>
        <Field name="GPUsPerNode" type="int">1</Field>
        <Field name="XArguments" type="string"></Field>
        <Field name="launchXServers" type="bool">false</Field>
        <Field name="XDisplay" type="string">:%l</Field>
        <Field name="numThreads" type="int">0</Field>
        <Field name="constrainNodeProcs" type="bool">false</Field>
        <Field name="allowableNodes" type="intVector"></Field>
        <Field name="allowableProcs" type="intVector"></Field>
        <Field name="profileName" type="string">preproc</Field>
    </Object>
    <Field name="activeProfile" type="int">2</Field>
    </Object>

.. code-block:: xml
   :caption: Host and launch profile template for Fram

    <?xml version="1.0"?>
    <Object name="MachineProfile">
    <Field name="hostNickname" type="string">fram.sigma2.no</Field>
    <Field name="host" type="string">fram.sigma2.no</Field>
    <Field name="userName" type="string">marskar</Field>
    <Field name="hostAliases" type="string"></Field>
    <Field name="directory" type="string">/cluster/software/VisIt/2.13.0-intel-2017a</Field>
    <Field name="shareOneBatchJob" type="bool">false</Field>
    <Field name="sshPortSpecified" type="bool">false</Field>
    <Field name="sshPort" type="int">0</Field>
    <Field name="sshCommandSpecified" type="bool">false</Field>
    <Field name="sshCommand" type="stringVector">"ssh" "-X" </Field>
    <Field name="useGateway" type="bool">false</Field>
    <Field name="gatewayHost" type="string"></Field>
    <Field name="clientHostDetermination" type="string">MachineName</Field>
    <Field name="manualClientHostName" type="string"></Field>
    <Field name="tunnelSSH" type="bool">true</Field>
    <Field name="maximumNodesValid" type="bool">false</Field>
    <Field name="maximumNodes" type="int">1</Field>
    <Field name="maximumProcessorsValid" type="bool">false</Field>
    <Field name="maximumProcessors" type="int">1</Field>
    <Object name="LaunchProfile">
        <Field name="timeout" type="int">480</Field>
        <Field name="numProcessors" type="int">32</Field>
        <Field name="numNodesSet" type="bool">true</Field>
        <Field name="numNodes" type="int">1</Field>
        <Field name="partitionSet" type="bool">false</Field>
        <Field name="partition" type="string"></Field>
        <Field name="bankSet" type="bool">true</Field>
        <Field name="bank" type="string">nn9636k</Field>
        <Field name="timeLimitSet" type="bool">true</Field>
        <Field name="timeLimit" type="string">00:30:00</Field>
        <Field name="launchMethodSet" type="bool">true</Field>
        <Field name="launchMethod" type="string">srun</Field>
        <Field name="forceStatic" type="bool">true</Field>
        <Field name="forceDynamic" type="bool">false</Field>
        <Field name="active" type="bool">false</Field>
        <Field name="arguments" type="stringVector"></Field>
        <Field name="parallel" type="bool">true</Field>
        <Field name="launchArgsSet" type="bool">true</Field>
        <Field name="launchArgs" type="string">"--account=nn9636k --qos=preproc"</Field>
        <Field name="sublaunchArgsSet" type="bool">false</Field>
        <Field name="sublaunchArgs" type="string"></Field>
        <Field name="sublaunchPreCmdSet" type="bool">false</Field>
        <Field name="sublaunchPreCmd" type="string"></Field>
        <Field name="sublaunchPostCmdSet" type="bool">false</Field>
        <Field name="sublaunchPostCmd" type="string"></Field>
        <Field name="machinefileSet" type="bool">false</Field>
        <Field name="machinefile" type="string"></Field>
        <Field name="visitSetsUpEnv" type="bool">false</Field>
        <Field name="canDoHWAccel" type="bool">false</Field>
        <Field name="GPUsPerNode" type="int">1</Field>
        <Field name="XArguments" type="string"></Field>
        <Field name="launchXServers" type="bool">false</Field>
        <Field name="XDisplay" type="string">:%l</Field>
        <Field name="numThreads" type="int">0</Field>
        <Field name="constrainNodeProcs" type="bool">false</Field>
        <Field name="allowableNodes" type="intVector"></Field>
        <Field name="allowableProcs" type="intVector"></Field>
        <Field name="profileName" type="string">preproc</Field>
    </Object>
    <Field name="activeProfile" type="int">0</Field>
    </Object>



.. _visit_host_config:
   
Setting up host profiles
************************

When setting up client-server visualization from scratch, the user must create a *Host profile* on the client (i.e., the user's local version of VisIt).
The host profile specifies how to launch VisIt on the server.

To set up a new host profile, first launch VisIt on the client and then navigate to ``Options -> Host profiles`` and fill in the following fields in the ``Host settings`` tab:

* ``Host nickname``. E.g. *betzy.sigma.no*.
* ``Remote host name``. E.g. *betzy.sigma.no*.
* ``Path to VisIt installation``. Run ``module disp VisIt/<version>`` to find it. 
* ``Username``
* ``Tunnel data connections through SSH`` should be checked.

The host profile can then be stored locally by ``Options -> Save settings``.
After setting up a host profile, the user will be able to launch VisIt on the server and connect to it through a local client.
Simply select ``File -> Open`` and specify the host when opening files.

.. _visit_launch_config:

Setting up launch profiles
**************************

Launch profiles determine how VisIt is launched on the server. 
After creating the host profile, launch profiles are added through the ``Launch profiles`` tab under each host profile.

1. Go to ``Launch profiles`` and press ``New profile``.
2. Next, under ``Parallel``, select the launch method (e.g., ``srun``), the number of nodes, time limit, and project account (usually in the form nnXXXXk).
3. Under the ``Advanced`` tab, one can add launcher arguments. For launching the job in the ``devel`` queue, for example, check the launcher argument tick box and add ``--qos=devel`` to the corresponding field.

   .. note::
   
      Occasionally, queue systems may require that the job account is added to launcher arguments. In this case the user must also add ``--account=nnXXXXk`` to the launcher argument field. 
   

Citation
--------

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).
