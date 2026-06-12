(nextflow-on-saga)=
# Nextflow on Saga

[Nextflow] is a workflow manager that simplifies running complex, multi-step computational pipelines on HPC systems. It supports container technologies like Singularity and integrates natively with SLURM schedulers.

This guide demonstrates how to configure and run Nextflow pipelines efficiently on the Saga cluster, particularly when using Singularity containers.

## Getting Started with Nextflow

To make Nextflow available in your environment, load the module before running any scripts:
```bash
module load Nextflow/25.10.2
```

Nextflow runs on the login node as a lightweight orchestrator: it reads the pipeline script, resolves configuration, and submits each process as an individual SLURM job. All heavy computation happens on the compute nodes allocated by SLURM. The login node performs no intensive work.

A typical Nextflow setup on Saga involves three components:

- A **pipeline script** (`.nf`) defining the processes and the workflow connecting them
- **Configuration files** (`.conf`) controlling resources, paths, and cluster-specific settings
- A **wrapper script** (`.sh`) that prepares the environment and launches Nextflow

## Walkthrough: A Complete Example Pipeline

The following sections build a minimal pipeline step by step, explaining the reasoning behind each decision before introducing the corresponding code.

### 1. Telling Nextflow to Use SLURM and Singularity (`nris.config`)

By default, Nextflow runs processes locally, meaning heavy computation would land directly on the login node. To run on Saga properly, two things must be configured:

- **SLURM executor:** Setting `executor = 'slurm'` causes Nextflow to submit each pipeline process as its own individual SLURM job. Without this, Nextflow falls back to the local executor and runs heavy processes directly on the login node.
- **Singularity bind mounts:** Containers need access to your input files and output directories. Setting `singularity.autoMounts = true` handles this automatically; without it, Nextflow will fail to mount the working directory.

The `nris.config` below handles both and also routes jobs to the correct partition (`normal`, `bigmem`, `hugemem`) based on the requested memory. It contains no project-specific values and can be reused across pipelines without modification.


```groovy
//nris.config

profiles {

  // Umbrella profile placeholder (for future clusters, e.g. Olivia)
  nris { }

  // Saga internal profile
  saga {

    // Header info and legacy nf-core limits
    params {
      config_profile_description = 'NRIS Saga cluster profile provided by nf-core/configs.'
      config_profile_contact     = 'support@nris.no'
      config_profile_url         = 'https://documentation.sigma2.no'

      // For older nf-core pipelines; align with resourceLimits below
      max_memory = 6040.GB
      max_cpus   = 64
      max_time   = 336.h   // 14 days
    }

    process {
      executor = 'slurm'

      // Upper bounds for nf-core auto-resubmission
      resourceLimits = [
        cpus  : 64,
        memory: 6040.GB,
        time  : 336.h      // 14 days
      ]

      // Auto-select partition by total memory and time
      // - normal: time ≤ 7 days AND memory ≤ 178.5 GiB
      // - bigmem: (time > 7 days AND memory ≤ 3021 GiB) OR (178.5 GiB < memory ≤ 3021 GiB)
      // - hugemem: memory > 3021 GiB
      queue = {
        def memGiB = task.memory ? task.memory.toGiga() : 0
        def tHours = task.time ? task.time.toHours() : 0

        if (memGiB > 3021) {
          'hugemem'
        } else if (tHours > 168 || memGiB > 178.5) {
          'bigmem'
        } else {
          'normal'
        }
      }

      // Retry behavior
      maxRetries = 2

      // Run tasks directly in Nextflow work dir (Lustre) to avoid ESPIPE on /tmp
      // beforeScript = 'export TMPDIR=/localscratch/$SLURM_JOB_ID'
      // scratch = true

      // Project account: pass with --account
      clusterOptions = { params.account ? "--account=${params.account}" : '' }
    }


    singularity {
      enabled    = true
      autoMounts = true
      cacheDir   = "/cluster/work/users/${System.env.USER}/singularity_cache"
      tmpDir     = "/cluster/work/users/${System.env.USER}/singularity_tmp"
    }

    executor {
      queueSize       = 50
      submitRateLimit = '10/sec'
    }
  }
}

```

```{note}
Nextflow requires strict [Groovy formatting] for resources. Use `time=24.h` and `1.GB` instead of `time=24h` and `1GB`.
```

### 2. Configuring Per-Project Resources (`my_pipeline.conf`)

This is the only file you need to edit in order to run the example pipeline: set `account` at the top to your project account number.

Two Saga-specific issues need addressing in your project configuration:

**The `/tmp` limit:** Node temporary directories on Saga are limited to 17 GB. Memory-intensive steps will easily overflow this, causing `No space left on device` errors. The solution is to route `TMPDIR` to your persistent workspace instead.

**Login-node variable evaluation:** Nextflow evaluates configuration on the login node *before* submitting jobs to SLURM. Shell variables like `$SCRATCH` therefore expand to `null` at submission time. Use `System.getenv('USER')` with explicit absolute paths instead.

```{note}
Always ensure your final output directory (`params.outdir`) points to persistent storage (like `/cluster/projects/...`). Do not use `$SCRATCH` for final outputs, or they will be deleted when the job ends.
```

Below is an example `my_pipeline.conf` that addresses both issues:

```groovy
// my_pipeline.conf

// Replace with your project account number
def account = "<your_account>"

env.TMPDIR = "/cluster/work/users/${System.getenv('USER')}/tmp"

params {
    threads = 2
    outdir  = "/cluster/projects/${account}/my_results"
}

process {
    withName: qualityControl {
        memory = 2.GB
        time = 1.h
        cpus = 1
    }

    withName: dereplicateSequences {
        memory = 4.GB 
        time = 1.h
        cpus = params.threads
    }
}

singularity {
    runOptions = "--bind /cluster/projects/${account},/cluster/work/users"
}

report {
    enabled = true
    file    = 'execution_report.html'
    overwrite = true
}
```

```{note}
The `report` block instructs Nextflow to automatically generate `execution_report.html` when the pipeline finishes. This report is essential for checking the actual peak memory (`peak_rss`) used by each task, allowing you to optimize your resource requests for future runs.
```

### 3. Defining the Pipeline (`bio_pipeline.nf`)

A Nextflow pipeline is built from **processes** (individual computational steps, each running inside its own container) connected by a **workflow** block. Nextflow tracks the outputs of each process and passes them as inputs to the next, so execution order is determined by data dependencies rather than script order.

```groovy
// bio_pipeline.nf
nextflow.enable.dsl=2

process qualityControl {
    container 'staphb/fastqc:latest'
    publishDir params.outdir, mode: 'copy'
    output: path "qc_done.txt"
    script: """echo "Running Quality Control..." > qc_done.txt"""
}

process dereplicateSequences {
    container 'maestsi/metontiime:latest'
    publishDir params.outdir, mode: 'copy'
    input: path qc_status
    output: path "derep_done.txt"
    script: """echo "Running heavy sequence dereplication..." > derep_done.txt"""
}

workflow {
    qc_out = qualityControl()
    dereplicateSequences(qc_out)
}

```

```{note}
The container images in this example use `:latest` tags for simplicity. For production pipelines, pin images to a specific version tag or digest (e.g., `staphb/fastqc:0.12.1`) to ensure reproducible results across runs.
```

### 4. Launching Safely with a Wrapper Script (`run_pipeline.sh`)

Because Nextflow evaluates configuration on the login node before any SLURM jobs are submitted, certain runtime directories, such as `TMPDIR` and the Singularity cache, must already exist at launch time. A wrapper script is a convenient place to create them before handing off to Nextflow.

```bash
#!/bin/bash
set -e

PROJECT_ID="<your_account>"

module reset
module load Nextflow/25.10.2

echo "Creating missing directories..."
mkdir -p /cluster/projects/$PROJECT_ID/my_results
mkdir -p /cluster/work/users/$USER/tmp
mkdir -p /cluster/work/users/$USER/singularity_cache
mkdir -p /cluster/work/users/$USER/singularity_tmp

echo "Launching Nextflow..."
nextflow -c nris.config \
         -c my_pipeline.conf \
         run bio_pipeline.nf \
         -profile saga \
         --account $PROJECT_ID \
         -work-dir /cluster/projects/$PROJECT_ID/nfworkdir/ \
         -resume
```

```{note}
Including the `-resume` flag ensures that if the job fails and needs debugging, the job will resume instead of starting from scratch when rerunning.
```

## Best Practices for Running Pipelines

It is highly recommended to use `tmux` or `screen` when running pipelines. Without them, closing your terminal or losing your SSH connection will kill the Nextflow orchestrator process. Even though the currently running SLURM jobs may finish, Nextflow will no longer be alive to collect their results, submit the next steps of the pipeline, or handle failures.

Below are instructions on how to use `tmux` and `screen`:

`````{tabs}

````{group-tab} tmux
Start a new session:
```bash
tmux new -s <your_job>
```

To safely detach from the session while leaving Nextflow running in the background:

1. Press `Ctrl+B`.
2. Press `D` (for detach).

To reattach to the session:
```bash
tmux attach -t <your_job>
```
````

````{group-tab} screen

Start a new session:
```bash
screen -S <your_job>
```

To safely detach from the session while leaving Nextflow running in the background:

1. Press `Ctrl+A`.
2. Press `D` (for detach).

To reattach to the session:
```bash
screen -r <your_job>
```
````
`````

### Execute the Pipeline

Make the wrapper script executable:

```bash
chmod +x run_pipeline.sh
```

Open a `tmux` or `screen` session as described above, then run the pipeline:

```bash
./run_pipeline.sh
```

(review-the-execution-report)=
### Review the Execution Report

To open the execution report, open a terminal on your local machine and download the report:

```bash
scp <your_username>@saga.sigma2.no:/path/to/your/report/execution_report.html ~/Downloads/
```

Locate the downloaded `execution_report.html` on your computer and double-click it. It will automatically open in your default web browser (like Chrome, Firefox, Safari, or Edge).

## Profiling Memory Usage

If a process continually crashes with `Out Of Memory` errors, do not blindly increase the memory. Instead, temporarily force the specific process onto the `hugemem` partition by requesting a massive ceiling (e.g., `memory = 3000.GB`).

Once the job successfully completes, you can check how much memory it actually used in the `execution_report.html` (see [Review the Execution Report](#review-the-execution-report)), which logs the peak memory (`peak_rss`) for every task. Alternatively, run `seff <jobid>` on the completed job.

To find the Slurm id for your job, run

```bash
nextflow log
```

Pick the name of the run you want to check, and use this command to print the Slurm Job IDs:
```bash
nextflow log <run_name> -f name,status,native_id
```

Now you can run
```bash
seff <job-id>
```

Pay special attention to "Memory Utilized". You can update your configuration to request exactly that amount (plus a small buffer, for example 20%) for future runs.

## Troubleshooting Containerized Scripts

The minimal example in this guide uses simple shell commands inside each process. In real pipelines, process scripts are often separate files. This could for example be a Python script (`analyze.py`) that relies on internal dependencies like `Rscript`.

If you need to manually rerun or test such a script outside Nextflow, a common mistake is to use `singularity shell` and then run the script as a separate command:

```bash
# This does NOT work as intended in a bash script
singularity shell /path/to/container.sif
python /path/to/scripts/analyze.py   # runs outside the container
```

`singularity shell` opens an interactive shell *inside* the container and then exits. Any commands written after it on separate lines run outside the container, where dependencies like `Rscript` are not installed. The script will fail silently: exit code 0, no output.

**The solution:** Use singularity exec. This command explicitly runs your script inside the container and then exits. Be sure to bind (`-B`) your current working directory (`$PWD`) so the container has permission to read your inputs and write your outputs:

```bash
# The correct way to manually test a script
singularity exec \
    -B "$PWD" \
    /cluster/work/users/$USER/containers/my_bio_env.sif \
    python bin/analyze.py --input my_input.txt --output results.txt
```

[Nextflow]: https://docs.seqera.io/nextflow/
[Groovy formatting]: https://docs.seqera.io/nextflow/reference/process#memory