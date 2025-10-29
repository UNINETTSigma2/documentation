(olivia-nird integration)=

# Data staging and data integration between NIRD and Olivia

This guide provides an overview of how Olivia’s I/O nodes integrate with NIRD Data Peak and Data Lake to support 
efficient data transfer, pre-processing, and post-processing workflows. It explains the purpose of these nodes, 
how to access them, and best practices for managing data between the two systems.

## What are the I/O Nodes?

Olivia includes five dedicated high-performance Input/Output (I/O) nodes specifically designed for data-intensive 
operations. These nodes act as a bridge between Olivia's own Lustre filesystem and NIRD filesystems, 
namely `/nird/datapeak` and `/nird/datalake`, enabling seamless data movement and management.

The I/O nodes are optimized for:

- Transferring large datasets between Olivia and NIRD
- Efficient handling of input data required for large-scale computations

By using these nodes for heavy data-lifting tasks, you help keep the main login and compute nodes free from heavy 
I/O activities, ensuring better performance for all users.

### Hardware at a Glance:
- **Nodes**: 5x HPE ProLiant DL385 Gen 11 (hostnames svc01 to svc05)
- **CPUs**: 2x AMD EPYC Genoa 9534 (128 cores per node)
- **Memory**: 512 GiB per node

## How to Access the dedicated I/O Nodes

You can connect to any of the I/O nodes directly from Olivia's login nodes using ssh.

   ```bash
   $ssh svc01
   ```
You can replace svc01 with the specific I/O node name you wish to access ( svc02, svc03, svc04, sc05)

## Accessing your NIRD data

The dedicated I/O nodes provide direct connections to the NIRD  filesystems (both datapeak and datalake).
Your NIRD project data is available in the following mounted directories on the I/O nodes:

- `/nird/datapeak`
- `/nird/datalake`

Inside these directories, you can navigate to your specific NIRD project folder, which is named after your 
NIRD project number (e.g., NSXXXXK). You can check if your user has a NIRD project membership by using the
 `groups` command. If it returns an ‘ns’ group, you should have access to this folder on NIRD.

If your group does not yet have a NIRD project, you/your PI should send an application for resource allocation 
on  NIRD Data Peak and/or NIRD Data Lake project through our [administration system](https://www.metacenter.no).

## Data Handling Scenarios

Three workflows are detailed below for managing input and output data between NIRD (Data Peak or Data Lake) 
Project Area and Olivia. Each scenario outlines the workflow steps and pros and cons of the workflow.

### Scenario 1: Manual Data Transfer

In this approach, the user login to the service node, manually copies input data from NIRD Data Peak/Data Lake 
project area  to the work directory on Olivia, runs the job, and then manually copies the required output data back to the specific NIRD  Data Peak/ Data Lake project directories.

#### Step 1: Log in to a Service Node
Connect to one of the service nodes using ssh from login node.
Each service node is identified as svc0x, where x can be any number between 1 and 5.
Eg: 

```bash
   $ssh svc01
   ```
#### Step 2: Copy Input Data from NIRD

Before transferring data, ensure that you have created a suitable input directory under your project’s work area on Olivia.

From Data Peak:

```bash
   rsync -avh --progress /nird/datapeak/NSxxxxK/input_data/ /cluster/work/projects/nnxxxxk/job_input/
   ```
From Data Lake:

```bash
   rsync -avh --progress /nird/datalake/NSxxxxK/input_data/ /cluster/work/projects/nnxxxxk/job_input/
   ```

- NSxxxxK represents your NIRD project number(on Datapeak or Datalake).
- nnxxxxk represents your compute project number on Olivia.
- The trailing / in the source path ensures the contents of the directory are copied, not the directory itself.

#### Step 3: Run Your Job from login nodes

- [Running Jobs](running-jobs)

#### Step 4: Copy Output Data Back to NIRD

When your job completes, copy the output results from your work directory back to your NIRD project directory.
Make sure to create a corresponding directory on NIRD before initiating the transfer.
Rememeber you have to login into a service node to execute the data transfer (step1).

To Data Peak:
```bash
   rsync -avh --progress /cluster/work/projects/nnxxxxk/job_output/ /nird/datapeak/NSxxxxK/results/
   ```
To Data Lake: 
```bash    
   rsync -avh --progress /cluster/work/projects/nnxxxxk/job_output/ /nird/datalake/NSxxxxK/results/
   ```
Pros:

- Simple and transparent workflow
- Full user control over data management

Cons:
 
- Requires manual effort
- High risk of human error
- No automation
- Inefficient for large datasets or batch jobs

The `rsync` command is recommended for efficient, resumable, and verifiable data transfer between NIRD and Olivia.
You can use `cp` as an alternative, but rsync is more reliable, especially for large datasets.
For more details on rsync usage, refer to the [File Transfer section](file-transfer).

### Scenario 2: Automated Staging (Slurm Script with Stage-In/Stage-Out)

In this scenario, the stage-in and stage-out scripts are used to automate data transfer t. The job script handles both input staging (before execution) and output staging (after execution).

- Link to the automated staging documentation to be updated

### Scenario 3: Direct Read from Compute Nodes

In this approach, jobs are executed directly on the compute nodes, accessing input data stored on NIRD without 
copying it to the local work directory. Both Datapeak and Datalake are mounted as read-only on the Olivia 
compute nodes, allowing users to read data directly but not write back to these locations.

After the job completes, the user must manually copy the output data from the compute node to their NIRD project 
area on either datapeak or datalake (follow the step4 in Scenario1).

#### Step1: Access Input Data Directly from

You can specify the path to your NIRD data directly in your job script or application.

#### Step2: Copy Output Data Back to NIRD

Follow the step4 in Scenario 1.

Pros:

- Simplified workflow (no pre-staging required)

Cons:

- Potentially slower performance when multiple jobs access the same storage simultaneously
- Risk of I/O bottlenecks
- No write access to NIRD from compute nodes

## Best Practices for Data Management

- Use the I/O nodes exclusively for data transfer and preparation tasks.
- Avoid running computationally intensive jobs on these nodes.
- Regularly transfer output data to NIRD Data Lake or NIRD Data Peak for mid-term storage.
- Remove unnecessary temporary data from Olivia to optimize storage usage.

## Long-Term Data Storage on NIRD

- If you do not yet have a NIRD project, you can apply for allocation on either NIRD Data Peak or NIRD Data Lake.
- All Olivia HPC projects are required to maintain a NIRD allocation for data management and storage.
- Output data stored temporarily on Olivia should be moved to NIRD Data Peak and/or NIRD Data Lake for mid-term preservation.
- Long-term data preservation can be achieved by depositing open access data into the NIRD Research Data Archive (NIRD RDA). 
- Please follow the [NIRD RDA documentation](research-data-archive) for more information.

