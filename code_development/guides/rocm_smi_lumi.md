---
orphan: true
---

(monitoring-gpus-on-lumi-g-with-rocm-smi)=
# Monitoring GPUs on LUMI-G with `rocm-smi`

To monitor GPUs on LUMI-G during the execution of your SLURM job, you can employ the `rocm-smi` command. This can be done by using `srun` with the `--overlap` option, which allows you to execute commands on the nodes allocated to your running job. Detailed information about using `--overlap` on LUMI-G is available [here](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/#using-srun-to-check-running-jobs).

## Steps to Monitor GPUs on jobs with a Single Node

1. Run the following command with your job-id on your single node job-id:
    ```bash
    srun --overlap --pty --jobid=<jobid> rocm-smi --showuse # replace with your desired option
    ```

## Steps to Monitor GPUs with multiple nodes

To monitor the GPU usage on jobs with multiple nodes, you can use the following steps:

1. **Identify the Allocated Nodes:**
   First, find out which nodes are allocated to your job by using the following command, replacing `<jobid>` with the ID of your SLURM job:

   ```bash
   sacct --noheader -X -P -oNodeList --jobs=<jobid>
   ```

2. **Execute `rocm-smi`:**
   Once you have the node names (e.g., nid00XXXX), execute `rocm-smi` to monitor the GPU usage:

   ```bash
   srun --overlap --pty --jobid=<jobid> -w <node_name> rocm-smi --showuse # replace with your desired option
   ```

   Replace `<node_name>` with the actual node identifier.

> **Note:** For this to work, use the `#SBATCH --gpus-per-node` directive instead of the `#SBATCH --gpus` directive in your job-script for multiple node jobs.

## Adding GPU Monitoring to a Job Script on LUMI-G

Monitoring GPU usage on the LUMI-G cluster can provide you with valuable insights into the performance and efficiency of your GPU-accelerated applications. By integrating ROCm-SMI (Radeon Open Compute System Management Interface) into your SLURM job script, you can collect GPU utilization statistics throughout the runtime of your job. Follow these instructions to modify your existing job script to include GPU monitoring with `rocm-smi`.

### Script for Expanding Node Ranges

To monitor specific GPUs, we must first resolve the node range into individual node names. The following script, named `expand_nodes.sh`, will be used in the job script to accomplish this:

```{code-block} bash
---
linenos:
emphasize-lines: 4, 6, 7-8, 16-18
---
#!/bin/bash

# Function to expand the node range like "nid[005252-005254]" into individual nodes
expand_node_range() {
    local node_range=$1
    if [[ "$node_range" == *"["* ]]; then
        local prefix=${node_range%%[*]}          # Extract the prefix ending at the first '['
        local range_numbers=${node_range#*[}     # Extract the range numbers
        range_numbers=${range_numbers%]*}        # Remove the trailing ']'

        local IFS='-'
        read -r start end <<< "$range_numbers"   # Read the start and end numbers of the range

        # Use printf to generate the sequence with zero padding based on the width of the numbers
        local width=${#start}
        for (( i=10#$start; i <= 10#$end; i++ )); do
            echo $(printf "nid%0${width}d" $i)
        done
    else
        echo "$node_range"
    fi
}
# Check if an argument was provided
if [ $# -eq 1 ]; then
    # Call the function with the provided argument
    expand_node_range "$1"
else
    echo "Usage: $0 <node_range>"
    exit 1
fi
```
Key elements of the expand_nodes.sh script:

1. The `expand_node_range` function (line 4) takes a string representing a range of nodes and expands it to individual nodes.
2. Checks for the presence of "[" to determine if it's a range (line 6).
3. Extracts the prefix and range numbers (lines 7-8).
4. Uses a for loop (lines 16-18) to iterate through the range and generate node names with proper zero padding.

Be sure to make the script executable before attempting to use it in your job script:

```{code-block} bash
chmod +x expand_nodes.sh
```


### Modified Job Script with GPU Monitoring

The following job script, `monitored_job_script.sh`, has been enhanced to include GPU monitoring capabilities. The GPU monitoring is encapsulated within a function and is designed to run concurrently with the main job.

```{code-block} bash
---
linenos:
emphasize-lines: 16, 21, 23-26, 32-35
---
#!/bin/bash -e
#SBATCH --job-name=<name>
#SBATCH --account=project_4650000XX
#SBATCH --time=XX:XX:XX
#SBATCH --partition=standard-g # or dev-g
#SBATCH --nodes=<nr_nodes>
#SBATCH --ntasks-per-node=<nr_tasks_per_node>
#SBATCH --gpus=<nr_gpus>
#SBATCH --gpus-per-node=8 # Rocm-smi only works on full nodes
#SBATCH -o %x-%j.out

# Load necessary modules
# ...

# Define the GPU monitoring function
gpu_monitoring() {
    local node_name=$(hostname)
    local monitoring_file="gpu_monitoring_${SLURM_JOBID}_node_${node_name}.csv"

    echo "Monitoring GPUs on $node_name"
    rocm-smi --csv --showuse --showmemuse | head -n 1 > "$monitoring_file"

    while squeue -j ${SLURM_JOBID} &>/dev/null; do
        rocm-smi --csv --showuse --showmemuse | sed '1d;/^$/d' >> "$monitoring_file"
        sleep 30 # Change this value to adjust the monitoring frequency
    done
}

export -f gpu_monitoring

nodes_compressed="$(sacct --noheader -X -P -o NodeList --jobs=${SLURM_JOBID})"
nodes="$(./expand_nodes.sh $nodes_compressed)"
for node in $nodes; do
  srun --overlap --jobid="${SLURM_JOBID}" -w "$node" bash -c 'gpu_monitoring' &
done

# Run the main job task
srun your_program

```

Key elements of the `monitored_job_script.sh` script:

1. We define a `gpu_monitoring` function (line 16) to capture GPU usage data.
2. The `--csv` flag in the `rocm-smi` command (line 21) is used to format the output as comma-separated values, making it easier to parse and analyze later.
3. The loop on lines 23-26 ensures that GPU data is captured at regular intervals until the job completes.
4. The function is exported (line 29) so that it can be called across different nodes within the job.
5. In lines 32-35 we expand the node range into individual nodes using the `expand_nodes.sh` script. Then we initiate the monitoring on each node in a loop using `srun`.

Note on ROCm-SMI flags:

- The `--showuse` and `--showmemuse` flags included with `rocm-smi` show GPU utilization and memory usage, respectively. These flags can be substituted or extended with other flags that are relevant to the specific monitoring requirements of your job. Using the `--csv` format ensures that the output is easily readable and can be processed with standard data analysis tools after the job has concluded.

### Submitting the Modified Job Script

To submit the job script with GPU monitoring enabled, use the following SLURM command:

```bash
sbatch monitored_job_script.sh
```

### Reviewing the Monitoring Data

Upon completion of your job, you can review the collected GPU usage and performance data. For each job, you will find a consolidated CSV file with a naming pattern of `gpu_monitoring_<jobid>_node_<nodename>.csv`. This file contains time-stamped metrics that will allow you to assess the GPU usage over the duration of the job.

Analyze the CSV data files using your preferred data processing tool to gain insights into the GPU resource utilization and identify potential bottlenecks or inefficiencies in your application's performance.

Note to Users: The provided scripts for GPU monitoring serve as an adaptable framework. Depending on the specific requirements of your computation workload, you may need to modify the scripts to fit your needs. Adjustments may include changing the frequency of data capture, modifying the captured metrics, or altering how the node expansion is handled. Use the scripts as a starting point and tailor them to surmount the individual challenges associated with monitoring in a HPC environment like LUMI-G.
