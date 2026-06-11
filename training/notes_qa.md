# Frequently Asked Questions from training events

Every training event hosted by NRIS has an associated collaborative document where participants can ask questions and make comments in real time during the course. A fully anonymized version of these documents is made available to the public after the event. 

Additionally, we compile the questions and answers from these documents into a frequently asked questions (FAQ) list for easy reference.

## Frequently Asked Questions (FAQ)
### Terminal and UNIX basics

**Q: Why do we use terminal/command line instead of apps/GUI?**

**A:** Because typing commands is a fast and flexible way to use a computer.

On high-performance computing systems it is especially useful because it works well over remote connections, and because the same commands work on login nodes and inside job scripts.

This is also useful on your own laptop: you can repeat tasks without lots of clicking, keep a clear history of what you did, and combine small tools to solve bigger tasks.

**Q: What is a terminal, a shell, and the “prompt”?**

**A:** The terminal is the window where you type commands.

The shell is the program that reads what you type and runs it. On our systems the most common shell is `bash` (the “Bourne Again SHell”). `bash` is what makes commands like `cd`, `ls`, and `grep` work, and it is also what runs your shell scripts.

The prompt is the text you see before you type (often ending in `$`). It usually shows your username, which machine you are on, and your current folder. For example: `[USERNAME@login-1.saga ~]$` means you are logged in as user `USERNAME` on the machine `saga` on the login node `login-1` and you are in your home folder (`~`).

**Q: What do `pwd`, `ls`, `cd` do?**

**A:**
- `pwd` prints the path of the folder you are currently in.
- `ls` lists files and folders.
- `cd` changes to a different folder.

Tip: use tab completion to avoid typing long names.

How to use tab completion:
- Start typing a command or a path, then press the <kbd>Tab</kbd> key.
- If there is only one match, it will complete the name for you.
- If there are multiple matches, press <kbd>Tab</kbd> again to show the options.

This is very useful to reduce typing mistakes and speed up your work.


**Q: What is `~`? What is `$HOME`?**

**A:** Both refer to your home directory (your personal starting folder on the cluster). This is your personal storage space that only you can see and access (unless you explicitly give someone else permission).

- `~` is a shortcut for your home directory.
- `$HOME` is the same home directory written as a variable that many programs and scripts use.

Tip: The command `cd` with no arguments takes you back to your home directory.

**Q: What is the difference between single quotes `'...'` and double quotes `"..."`?**

**A:** In most day-to-day use:

- Use quotes when you have spaces in something, like a folder name. E.g. if you want to access a folder called `my files` (note the space), you need to quote it: `cd "my files"` or `cd 'my files'`. If you just type `cd my files`, the shell will interpret `my` and `files` as two separate things.

- Single quotes `'like this'` keep the text exactly as written.
- Double quotes `"like this"` also keep spaces together, and they also allow *variables* like `$HOME` to turn into your home directory.

When in doubt, it is often safe to use double quotes.

**Q: What is the difference between a pipe `|` and redirect `>` / `>>`?**

**A:** They both deal with “where output should go”, but they do different things:

- A redirect writes output into a file.
  - `>` means “write to this file (and replace what was there before)”.
    - Example: `ls > files.txt` saves the list of files into `files.txt`.
  - `>>` means “add to the end of this file”.
    - Example: `echo "one more line" >> notes.txt` adds a new line to `notes.txt`.

- A pipe sends output into another command.
  - `|` means “take the output from the left command and use it as input to the right command”.
    - Example: `ls | grep ".txt"` lists files, then filters to only show lines containing `.txt`.

Rule of thumb:
- Use `>` / `>>` when you want to save output to a file.
- Use `|` when you want to process output with another command.

**Q: Why did `less` show “garbage” or lots of numbers?**

**A:** You likely opened a binary (non-text) file. Use `file <name>` to check file type before viewing. `less` is best for long text files; `cat` is fine for short ones.

**Q: How do I stop a command that is running too long?**

**A:** Use <kbd>Ctrl</kbd>+<kbd>C</kbd> to interrupt the running command.

### Moving/creating/renaming/deleting files

**Q: What happens if I run `mkdir` and the folder already exists?**

**A:** `mkdir <name>` creates a new folder. If a folder with that name already exists, `mkdir` stops and prints an error.

If you want a “safe” version that does not error when the folder already exists, use:

- `mkdir -p <dir>`

`-p` also creates any missing parent folders, so if you want to create nested folders inside one another `nested/folder/structure`, you can do it in one step with `mkdir -p nested/folder/structure`.
```
nested/
└── folder/
    └── structure/
```

**Q: Why doesn’t `mv *.output *.out` work to rename many files?**

**A:** `*.output` *expands* to many filenames by replacing the wildcard character `*` with all files matching the pattern. However, bash can not expand the second wildcard because these are not files that exist yet. For example, say you are in a folder with these files:
```
folder/
├── file1.output
├── file2.output
├── file.txt
└── image.jpg
```
Here, the wildcarded expression `*.output` will resolve to `file1.output file2.output`. However, `*.out` will not resolve into any *currently existing* files, so the shell can not guess what you are trying to do.

For many files, you usually use a small loop. For example:
```bash
for FILENAME in *.output; do
  mv "$FILENAME" "${FILENAME%.output}.out"
done
```

**Q: Is `mv -r` a way to rename many files?**

**A:** No. `mv` does not have a recursive rename option.

If you need to rename many files, do it explicitly (often with a loop, sometimes with `find`). If you’re unsure, ask for help before running a bulk rename.

**Q: How do I avoid accidentally overwriting files?**

**A:** Two simple habits help a lot:

- Check what a wildcard expands to before you run a destructive command. Example: run `ls *.txt` before `rm *.txt`.
- Be careful with output redirects (`>`): they can replace a file.

If it’s important data, keep it backed up somewhere safe whenever possible.

**Q: Is there an alternative to `rm`?**

**A:** On most HPC systems, `rm` is permanent.

If you are not 100% sure, a simple safer workflow is:

- Create a “trash” folder (for example `~/trash`) and move files there first.
- Empty it later when you are sure.

For important work, rely on backups and/or version control rather than “I can undo it later”.

### Finding files/folders

**Q: How can I find files without manually browsing every folder?**

**A:** Use `find` to search recursively.

Example (find all files ending in `.out` under the current folder):

- `find . -type f -name "*.out"`

Start from a specific folder (`.` for “here”, or a project folder). Searching a smaller area is faster and avoids accidental heavy scans.

**Q: Why shouldn’t I run `find / ...`?**

**A:** `/` means “the entire system”. That scan can be very slow and will visit many places you do not need (and often cannot access).

Prefer searching inside your own folders (home or work or project folders).

**Q: How do I search for text inside files?**

**A:** Use `grep`.

Example (search for the text `ERROR` inside files under the current folder):

- `grep -R "ERROR" .`

If you get too many hits, narrow the search to a folder or to certain file types.

**Q: Why do quotes matter when searching?**

**A:** Quotes keep things together.

For example, `grep -R "hello world" .` searches for the two-word phrase. Without quotes, the shell would treat `hello` and `world` as separate arguments.

### File/folder permissions

**Q: Can other users read my files in my home directory?**

**A:** No.

Your home directory is private by default. But permissions can be changed (by you), and shared folders (projects and shared areas) are meant for collaboration.

If you are unsure, check with `ls -ld <folder>` (for a folder) and `ls -l <file>` (for a file).

**Q: What do `r`, `w`, `x` mean in `ls -l` output?**

**A:**
- `r` (read): you can view the contents
- `w` (write): you can change the contents
- `x` (execute): you can run it as a program

They are shown separately for: the file owner, the file’s group, and “others” (everyone else).

For folders, `x` roughly means “you are allowed to enter this folder”.

**Q: What is the “group” and why does it matter in project folders?**

**A:** A “group” is a named set of users (for example, everyone in a project).

Project/team folders often rely on group permissions so that:
- everyone in the project can read shared files
- selected people can write

If a file gets the wrong group or restrictive permissions, your collaborators may not be able to read or edit it until it is changed.

### Remote login and SSH

**Q: How do I check if SSH is installed and working?**

**A:** Run `ssh -V`. If you see a version number, SSH is installed. In almost every case, which version of SSH you have does not matter.

**Q: When I log in to a cluster sometimes I see `login-2` and other times something like `login-5`. Is that a problem?**

**A:** No. Clusters often have several login nodes.

You will be sent to different ones for load balancing. This is normal. You still have access to the same files and environment no matter which login node you land on.

**Q: Are home directories the same on all login nodes?**

**A:** Yes (within the same cluster). Your home directory is the same on all login nodes.

Different clusters are separate systems, so they do not share the same home directory.

**Q: How do I use VS Code to connect via SSH?**

**A:** Install the VS Code “Remote - SSH” extension.

Then:
- Connect to the cluster over SSH
- Open a folder on the remote system (often your home directory, or your project folder)

If VS Code has problems, you can always fall back to a normal terminal with `ssh`.

See [Connecting to a system with Visual Studio Code](/code_development/guides/vs_code/connect_to_server.md) and [SSH](/getting_started/ssh.md) for more details.

**Q: Why does VS Code sometimes use lots of space in my home directory?**

**A:** VS Code Remote installs helper files on the remote system (often under `.vscode-server` in your home directory).

These files can grow over time and count towards your home quota.

See [Storage quota](/files_storage/quota.md) for how to check what is using space.

**Q: How does two-factor authentication (2FA) affect SSH?**

**A:** If 2FA is enabled, normal SSH login and file transfer will require an extra one-time code.

For repeated file transfers, there is a documented exception: if you set up SSH keys, you can use SFTP or, with recent OpenSSH versions, `scp` over port `12` without entering an OTP for every transfer.

This port `12` method is only available on Forskningsnett networks, so when working from home you may need VPN to your institution first.

See [One-time-pad (OTP) / Two-factor authentication](/getting_help/two_factor_authentication.md) and [SSH](/getting_started/ssh.md) for the full setup and limitations.

### Login nodes vs compute nodes

**Q: What is the difference between login nodes and compute nodes?**

**A:**
- Login nodes: where you log in. Use them for editing files, setting up software, and submitting jobs.
- Compute nodes: where your actual heavy work runs (calculations, training, simulations). You get access to compute nodes through the scheduler by starting a job.

**Q: What should I not do on login nodes?**

**A:** Avoid heavy work:
- long-running programs
- CPU/GPU-intensive computations
- many parallel tasks

Use the scheduler for that work so it runs on compute nodes.

**Q: Can I compile code on login nodes?**

**A:** Small builds are usually fine on login nodes.

If the build is large (takes a long time, uses many cores, or uses `make -j`), do it in an interactive job or batch job on compute nodes.

**Q: Why can’t I “just SSH into a compute node”?**

**A:** Compute nodes are shared resources. The scheduler decides who gets which node, and when.

You access compute nodes by:
- starting an interactive job (for example `salloc`), or
- submitting a batch job (for example `sbatch`).

**Q: Why can’t I access my tmux/screen session from another login node?**

**A:** A `tmux`/`screen` session lives on the machine where you started it.

If you log in to a different login node later, that session is not running there. You can login to a specific login node to reconnect by SSHing to it directly (for example `ssh USERNAME@login-2.saga.sigma2.no`).

**Q: When I log in, am I “in” a particular project/account?**

**A:** No. SSH login is tied to your user account, not to one specific project.

You choose the project/account when you submit jobs, usually with `--account` or `#SBATCH --account`. Use the `projects` command to see which projects you are a member of, and `cost` or `cost -d` to see quota and usage information.

See [Projects and accounting](/jobs/projects_accounting.md) for more details.

### Scheduling jobs with SLURM

**Q: Why do we need a scheduler like Slurm?**

**A:** Because many users share the same cluster.

Scheduler:
- decides which jobs run when
- gives jobs the resources they asked for (CPU, memory, GPUs)
- prevents one user from hogging all the resources

**Q: What is a partition, QoS, or reservation?**

**A:**
- Partition: A part of the cluster with specific resources and rules, for example the `accel` partition has GPUs available, while `bigmem` has nodes with lots of memory.
- QoS: settings that may change job scheduling priority or resource limits (for example, if you need to run small test jobs you can use `devel` QoS to get high priority in the queue).
- Reservation: a special “reserved” set of resources, such as specific nodes on a machine at a specific time (often used for courses)

**Q: What are `#SBATCH` lines? Are they comments?**

**A:** They look like comments, but they are special lines that Slurm reads.

When you submit the script, Slurm scans the `#SBATCH` lines to learn what resources you want (time, memory, number of tasks, etc.).

**Q: Where does job output go? Why don’t I see it in my terminal?**

**A:** Your job runs “somewhere else” (on compute nodes), not in your current terminal.

The output from your job is written by Slurm to files. By default it looks like `slurm-<jobid>.out` in the folder where you submitted the job.

You can choose these filenames and locations of the output with the `--output` and `--error` Slurm commands.

**Q: How do I see my jobs and cancel them?**

**A:**
- List your jobs: `squeue --me`
- Cancel a job: `scancel <jobid>`
- Cancel all your jobs: `scancel --me`

See [Monitoring jobs](/jobs/monitoring.md) for more commands and examples.

**Q: What do Slurm job states like `PD`, `R`, `CG`, and `CD` mean?**

**A:** These are short Slurm job-state codes.

- `PD`: pending, waiting to start
- `R`: running
- `CG`: completing, job is finishing and Slurm is cleaning up
- `CD`: completed

There are more states and possible reason codes, so if you see something unfamiliar, check the job-state documentation.

See [Job States](/jobs/monitoring/job_states.md).

**Q: How do I see finished jobs or jobs that disappeared quickly from `squeue`?**

**A:** `squeue` only shows jobs that are queued, starting, or running.

If the job already finished, use `sacct -j <jobid>`. While a job is still running, `sstat -j <jobid>` can show usage information. If a short job finished almost immediately, also inspect the output file, which by default is called `slurm-<jobid>.out` and located in the submission directory.

See [Monitoring jobs](/jobs/monitoring.md).

**Q: Why use `--job-name` if Slurm already gives the job a job ID?**

**A:** The job ID is unique, but the job name is easier for humans to recognize.

It appears in tools like `squeue`, and it helps when you have many similar jobs. You can also use `--output` and `--error` to create more descriptive filenames, for example `slurm-%x-%j.out`, where `%x` is the job name and `%j` is the job ID.

See [Slurm Parameter and Settings](/jobs/job_scripts/slurm_parameter.md).

**Q: What happens if I underestimate `--time`?**

**A:** Slurm will stop (kill) the job when it reaches the requested time limit. If your job does not "checkpoint" progress periodically, you will lose all progress and all output if this happens to your job.

Start with a small test run, then increase `--time` based on what you observe. It is OK to ask for a bit more time than you *think* the job will need, you will only be billed for time actually used. If you ask for 12 hours and the job finishes after 10 and a half, the cost will only be for the 10 and a half. However, be considerate and do not ask for 6 days if you only need 2 hours, as this will lead to longer queue times for everyone.

For long jobs you should set up checkpointing so that if your job crashes or exits, you do not lose all progress.

**Q: What do `--ntasks`, `--nodes`, and `--cpus-per-task` mean?**

**A:** This depends on how your program runs in parallel, but a common mental model is:

- `--ntasks`: how many separate processes you want
- `--cpus-per-task`: how many CPU cores each process can use (threads)
- `--nodes`: how many machines the job will use

If you are unsure, start simple (1 task) and follow the application’s documentation.

**Q: I got “Invalid account or account/partition combination specified”. What does it mean?**

**A:** It usually means your job is not connected to a valid project/account for that partition.

Check that:
- you used the correct `--account` (project)
- the partition/QoS you chose is allowed for that project

Run `projects` to see which accounts you have access to and `cost` to see the current usage and quotas associated with each account.

See [Projects and accounting](/jobs/projects_accounting.md) for more details.

**Q: Are interactive jobs billed differently than batch jobs?**

**A:** No. Interactive jobs still reserve resources, so they count like batch jobs.

### Modules and software

**Q: What happens when I run `module load`?**

**A:** It changes your environment so the software becomes available.

In practice, it adjusts things like `PATH` (where the shell looks for programs) and library paths so the software version(s) you loaded is used.

Note: Never run `module load` inside `.bashrc` or `.bash_profile` because this will inevitably lead to almost unexplainable errors down the line. Always load modules explicitly in your terminal session or inside job scripts when you need them.

**Q: How do I undo a module load?**

**A:**
- Remove one module: `module unload <name>`
- Reset everything: `module purge`

**Q: What is the difference between `module reset` and `module purge`?**

**A:** `module reset` returns your environment to the system defaults, while `module purge` unloads everything *it can*.

On NRIS systems, `module reset` and `module purge` both get you back to the default base environment. Because the standard environment module `StdEnv` is “sticky” and will not be unloaded unless you use `module purge --force`, both commands have the same effect.

See [The module system](/software/modulescheme.md).

**Q: Is the module environment shared between users or sessions?**

**A:** No. Modules affect only your current terminal session.

If you log out and log in again, you start fresh. Modules loaded also have no effect on the jobs you submit. You have to load the relevant modules *inside your job script* to use them in the job.

**Q: How do I find available software?**

**A:** Common commands:

- `module avail` (browse what exists)
- `module spider <name>` (search; often more useful)

On Olivia, also check [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md), because the recommended workflow differs from the older clusters.

**Q: How do I know the command name after loading a long module name?**

**A:** The module name is not necessarily the command you run.

After loading, you typically use the program’s normal command (for example `python`, `R`, `gmx`, `lmp`, etc.).

If you are unsure:
- Try `which <command>`
- Check `module show <module>` for hints

**Q: What is a toolchain (e.g., GCCcore/foss/gompi) and why do modules “swap” compilers?**

**A:** A toolchain is a matching set of building blocks (compiler, MPI, math libraries) that software is built against.

Many HPC applications only work correctly with the toolchain they were built with. So when you load a module, the system may automatically load (or swap) the required compiler/MPI modules.

**Q: What is `StdEnv` and why is it sticky?**

**A:** It sets up the cluster’s “standard environment” (default module paths and assumptions).

It is “sticky” because many other modules depend on it, and it is designed to always be loaded. If you accidentally mess up the module environment, purging all loaded (non-sticky) modules with `module purge` usually restores a working state. If that does not help, logging out and back in again will restore the default state.

### Compiling/installing code

**Q: Where should I install my own software (Python/R packages, libraries)?**

**A:** First, check whether the software is already available as a maintained module. That is the preferred option.

If you need a custom environment, prefer container-based approaches next, especially on Olivia: use Apptainer or Olivia's HPC-container-wrapper rather than native `pip` or `conda` installs.

Only use `pip`, virtual environments, or Conda on systems where that workflow is allowed, and only when there is no suitable module or container workflow. On Olivia, native `pip` and Conda installs are not allowed; use HPC-container-wrapper or Apptainer instead. The reason is not just `$HOME` quota: `pip` and Conda create very large numbers of small files, and that stresses the shared filesystem even if the environment lives outside `$HOME`.

If you must install it yourself, keep it out of `$HOME`. Use cluster project/work areas for shared software, configs, staging, and active working files close to jobs. Put genuinely large data on NIRD instead.

See [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md), [Containers on NRIS HPC systems](/code_development/guides/containers.md), [Installing software with Conda](/software/userinstallsw/conda.md), and [Storage areas on HPC clusters](/files_storage/clusters.md).

**Q: Should I install from login node or compute node?**

**A:** Many installs can be done on login nodes.

If an install/compile is heavy (takes a long time or uses many cores), do it in an interactive job or a short batch job.

If you need special hardware (like GPUs), you must install/build on a node that has that hardware.

**Q: Why do my Python packages “disappear” when I load a different Python module?**

**A:** Because Python packages are installed into a specific Python interpreter (or environment).

If you load a different Python module, you are using a different interpreter, and it will not see the packages installed for the old one.

Rule of thumb: load the Python module you plan to use first, then create/install into an environment for that Python.

**Q: Can I share environments with my group?**

**A:** Yes. The preferred thing to share is the definition of the environment, not a raw `pip` or Conda tree.

- Best: share a maintained module name, an Apptainer image, or an HPC-container-wrapper setup.
- Next best: share `environment.yml` for Conda or `requirements.txt` for pip/venv so others can recreate the environment.
- Shared direct installs in a project directory can work, but they are more fragile. On Olivia native `pip` and Conda installs are not allowed.

See [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md), [Installing software with Conda](/software/userinstallsw/conda.md), and [Installing Python packages](/software/userinstallsw/python.md).

### Containers

**Q: Can I use Docker on the clusters?**

**A:** Not directly. Running the Docker daemon requires privileges that are not suitable on shared HPC systems. But the clusters are running Apptainer as a container runtime, and Apptainer can pull and run Docker images without needing Docker itself to run on the cluster.

See [Containers on NRIS HPC systems](/code_development/guides/containers.md) and [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md).

**Q: Can Apptainer run Docker images?**

**A:** Yes. Apptainer can frequently pull and run images from Docker registries.

Follow the cluster documentation for the recommended way to do this.

See [Containers on NRIS HPC systems](/code_development/guides/containers.md) and [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md).

**Q: When should I use containers?**

**A:** Usually when, on your laptop or PC, your next thought would be “I’ll just use `pip`” or “I’ll make a Conda environment”. On shared HPC systems, a container is often the better way to get that custom software stack without creating huge numbers of small files on the shared filesystem.

Containers are especially useful for Python/R environments, AI stacks, and tools with awkward dependencies. If a maintained module already exists and does what you need, that is still the simpler and better first choice. On Olivia, HPC-container-wrapper fills a similar role for many Python and R workflows.

See [Containers on NRIS HPC systems](/code_development/guides/containers.md) and [Installing and Using Software on Olivia](/hpc_machines/olivia/software_stack.md).

### Storage and quotas

**Q: Why is the home directory quota so small?**

**A:** Because `$HOME` is meant for small personal files and configuration, not for large datasets, outputs, or large software environments.

Any genuinely large files should usually live on NIRD Data Peak or Data Lake, and long-term published data should go to the NIRD Research Data Archive. Cluster project/work areas are better for shared software, configs, staging, and active working files close to jobs, not as the main home for bulk storage.

That keeps pressure off the shared cluster filesystem and leaves home space for what it is meant for.

See [Storage areas on HPC clusters](/files_storage/clusters.md), [NIRD storage overview](/files_storage/nird_lmd.md), [Storage quota](/files_storage/quota.md), and [NIRD Research Data Archive (NIRD RDA)](/nird_archive/user-guide.md).

**Q: What is `dusage` and what do “soft” vs “hard” quota mean?**

**A:**
- `dusage` shows how much storage (and sometimes how many files) you are using.
- Soft quota: you are over a warning limit (you may get a grace period).
- Hard quota: you cannot write more until you free space.

See [Storage quota](/files_storage/quota.md) for examples and troubleshooting.

**Q: Why can’t I create even an empty file when quota is full?**

**A:** Because “quota full” means “no more writes allowed”. Even an empty file needs filesystem metadata, so it still counts.

**Q: How do I find what uses space or too many files?**

**A:** Common beginner-friendly steps:

- Find large folders: `du -sh *` (run inside the directory you want to inspect)
- Count many files (example): `find . -type f | wc -l`

If you find a big directory, go into it and repeat until you locate what is using the space.

**Q: What is scratch (`$SCRATCH`) and localscratch (`$LOCALSCRATCH`)? Do I SSH there?**

**A:** `$SCRATCH` is temporary per-job storage, but the exact path depends on the system and sometimes the node type.

- On Betzy and Saga it is typically `/cluster/work/jobs/$SLURM_JOB_ID`.
- On Olivia it can be node-local `/localscratch/$SLURM_JOB_ID` on CPU nodes or `/cluster/software/gpujobscratch/jobs/$SLURM_JOB_ID` on GPU nodes.
- `$LOCALSCRATCH` is a separate variable only on Saga.

You do not “SSH to scratch”. Your Slurm job gets access to it on the compute node, and you should copy important output back to project or NIRD storage before the job ends.

See [Storage areas on HPC clusters](/files_storage/clusters.md) and [Local storage for scratch files](/jobs/scratchfiles.md).

### Data transfer

**Q: Is it OK to run `scp`/`rsync` from a login node?**

**A:** Yes. Login nodes are the right place for file transfers.

Compute nodes are for scheduled compute jobs and may have network restrictions.

See [Data transfer](/getting_started/data_transfer.md).

**Q: Why use `rsync` instead of `scp` or VSCode’s file browser?**

**A:**
- VSCode is convenient for small files.
- `scp` is simple for “copy once”.
- `rsync` is best for large directories because it can:
  - resume transfers if interrupted
  - only copy what changed

See [Data transfer](/getting_started/data_transfer.md) for examples and tool choices.

**Q: What should I use on Windows if `rsync` is not available?**

**A:** The closest match to the Linux/macOS examples is usually Windows Subsystem for Linux (WSL), because then `ssh`, `scp`, and `rsync` behave much like they do on the other platforms.

If you want a simpler drag-and-drop workflow, WinSCP is a common choice. VS Code Remote - SSH is also useful if your main goal is editing files rather than bulk transfer.

See [Data transfer](/getting_started/data_transfer.md), [SSH](/getting_started/ssh.md), and [Connecting to a system with Visual Studio Code](/code_development/guides/vs_code/connect_to_server.md).

**Q: Where should I transfer large data to on the cluster?**

**A:** Put large data in the storage area that matches its role:

- cluster project/work area for staging, active jobs, shared software, configs, and temporary processing
- NIRD Data Peak or Data Lake for larger shared datasets and persistent storage
- NIRD Research Data Archive for long-term open preservation

Avoid filling `$HOME` with large transfers unless they are genuinely small personal files.

See [Storage areas on HPC clusters](/files_storage/clusters.md), [Olivia and NIRD data workflows](/hpc_machines/olivia/olivia-nird.md), and [NIRD Research Data Archive (NIRD RDA)](/nird_archive/user-guide.md).

### Using shared resources responsibly

**Q: Why should I avoid requesting far more resources than I need?**

**A:** Because you reserve resources that nobody else can use while your job is running.

If you request far more than you need, your job may also wait longer in the queue, and your project/accounting may be charged for resources you never used. Note carefully: If your job requests 1000 CPUs and runs for 24 hours, you will be billed for 24,000 CPU hours, even if your job only *actually* used 10 CPUs. This is unlike compute time, where if you request 24 hours but your job finishes in 6 hours, you are only billed for 6 hours (of the resources you requested).

**Q: How do I choose a good number of cores/memory?**

**A:** Start with small tests:
- run a short job
- measure how much memory and CPU it actually uses
- scale up based on results and application documentation

**Q: Should I stage data transfers inside a big compute job?**

**A:** Usually no.

If you are moving large data between NIRD and the cluster, transfer it before/after the compute step, or on Olivia use Slurm stage-in/stage-out where that fits. That avoids spending CPU/GPU time mostly waiting on I/O, and it makes transfer failures easier to detect and recover from.

For small files, doing it inline is often fine.

See [Staging In / Out Files from / to NIRD](/jobs/job_scripts/stage_in_stage_out.md), [Olivia and NIRD data workflows](/hpc_machines/olivia/olivia-nird.md), and [Data transfer](/getting_started/data_transfer.md).

### How to get help

**Q: Where do I ask for help?**

**A:** Use the official support channels listed in the documentation.

To get help faster, include:
- which cluster/system you used
- what you tried
- the exact error message
- relevant job ID(s) and log files

See [Getting help](/getting_help/support_line.md).

**Q: What information should I include in a support request?**

**A:** The goal is that someone else can understand and reproduce the problem.

Include:
- which system you are using
- what you were trying to do (what is the ultimate goal you want to achieve, not necessarily just what is the specific command you ran that failed supposed to do)
- exact commands you ran (from when you log in to when the error happened)
- the full error message (copy/paste)
- your job script (if Slurm)
- job ID (if Slurm)
- where your input/output files are located

**Q: Can support staff log into my account?**

**A:** In general, you should never share your password with anyone.

The preferred approach is to:
- share the relevant log files, job scripts, and error messages
- if needed, share files via safe permissions or a designated shared location

If special access is required, support will tell you the approved procedure.


## Collaborative notes archives from past training events

### 2026
- {ref}`training-2026-spring-onboarding`: [Day 0](https://md.sigma2.no/hpc-onboarding-April-14-2026), [Day 1](https://md.sigma2.no/hpc-onboarding-April-15-2026), [Day 2](https://md.sigma2.no/hpc-onboarding-April-16-2026)

### 2025 
- {ref}`training-2025-autumn-onboarding`: [Day 0](https://md.sigma2.no/hpc-onboarding-October-14-2025), [Day 1](https://md.sigma2.no/hpc-onboarding-October-15-2025), [Day 2](https://md.sigma2.no/hpc-onboarding-October-16-2025)
- [Q&A from Best Practices and Tools for HPC Spring 2025 Episode 5: Apptainers on HPC: Usecases and Examples](https://md.sigma2.no/best-practices-and-tools-episode5)
- [Q&A from Best Practices and Tools for HPC Spring 2025 Episode 4: Advanced shell scripting and utilities part 1: advanced awk and sed](https://md.sigma2.no/best-practices-and-tools-episode4)
- [Q&A from Best Practices and Tools for HPC Spring 2025 Episode 3: Containers on Clusters](https://md.sigma2.no/best-practices-and-tools-episode3)
- [Q&A from Best Practices and Tools for HPC Spring 2025 Episode 2: Advanced shell scripting and utilities part 1](https://md.sigma2.no/best-practices-and-tools-episode2)
- [Q&A from Best Practices and Tools for HPC Spring 2025 Episode 1: How to parallelize independent tasks](https://md.sigma2.no/best-practices-and-tools-episode1)
- {ref}`training-2025-spring-onboarding`: [Day 0](https://md.sigma2.no/hpc-onboarding-May-06-2025), [Day 1](https://md.sigma2.no/hpc-onboarding-May-07-2025), [Day 2](https://md.sigma2.no/hpc-onboarding-May-08-2025)

### 2024
- [Q&A from Introduction to NIRD Toolkit on 24th October 2024](https://md.sigma2.no/nird-toolkit-training2024)
- {ref}`training-2024-autumn-onboarding`: [Day 0](https://md.sigma2.no/hpc-onboarding-Oct-15-2024), [Day 1](https://md.sigma2.no/hpc-onboarding-Oct-16-2024), [Day 2](https://md.sigma2.no/hpc-onboarding-Oct-17-2024)
- {ref}`training-2024-spring-scientific-computing`
  - [Q&A from NRIS Scientific Computing Workshop](https://md.sigma2.no/scientific-computing-workshop)
- {ref}`training-2024-spring-best-practices`: [Day1 (How to choose the right amount of memory and right number of cores)](https://md.sigma2.no/qanda-archive-best-practices-spring2024), [Day2 (Software Installatiion as user)](https://md.sigma2.no/qanda-archive-best-practices-spring2024-day2), [Day3 (File Transfer, Data Storage and NIRD)](https://md.sigma2.no/best-practices-training-spring2024)
- {ref}`training-2024-spring-onboarding`: [Day 0](https://md.sigma2.no/n1LZOKOmQR-clMgqgeOr4g), [Day 1](https://md.sigma2.no/weunEf2gT7mnM1okU4MOGA), [Day 2](https://md.sigma2.no/hpc-onboarding-April2024) 
- [Q&A from Introduction to FORTRAN Series Autumn 2023](https://md.sigma2.no/YQjft6kvR5SUolKJtm3QOA)
- [Q&A from Introduction to NIRD Toolkit on 24th October 2023](https://md.sigma2.no/introduction-to-nird-toolkit)

### 2023
- {ref}`training-2023-autumn-onboarding`: [Day 0](https://md.sigma2.no/ioTn-FGQTm2Bw3v5YeCXHQ), [Day 1](https://md.sigma2.no/2RZ9bF1BSaCpkQ90OuEXmA), [Day 2](https://md.sigma2.no/dEY7PA0ITkarb7jzW-XBWg)  
- {ref}`training-2023-spring-best-practices`: [Day 1](https://md.sigma2.no/3imgsU1KSV2SNk9m44kbnw), [Day 2](https://md.sigma2.no/fSq3uN77SnefUIEf5bP4vw), [Day 3](https://md.sigma2.no/Pm6B9ohCRl6Ne6VB26tkow) 
- {ref}`training-2023-spring-onboarding`: [Day 0](https://md.sigma2.no/45qiXesPSEWmdily6I65bA), [Day 1](https://md.sigma2.no/HauUyzY9QMWCJ5xKCpmVKQ), [Day 2](https://md.sigma2.no/yrGPYBTuQTa0SD50c6RhoQ)

### 2022
- {ref}`training-2022-autumn-best-practices`: [Day 1](https://md.sigma2.no/EZnXUmmsT9CGwlayz1F9xA), [Day 2](https://md.sigma2.no/So3XP6n0R56W42SnwqHoeg), [Day 3](https://md.sigma2.no/VLcxCXmNTdy4zKq_xaPUrA)
- {ref}`training-2022-autumn-onboarding`: [Day 0](https://md.sigma2.no/qqoDOErERAm2KIPMQKCKcQ), [Day 1](https://md.sigma2.no/XPgMoPiaRNeN-tLCYla3Ug), [Day 2](https://md.sigma2.no/3xb-QvL-RQ6g2-bQtavANQ)
- {ref}`training-2022-spring-best-practices`: [Day 1](https://md.sigma2.no/d2sIqLq1R8WTh8c6KJHg9A), [Day 2](https://md.sigma2.no/8a2nNrqmRJGpwepBejcv_A)
- {ref}`training-2022-spring-onboarding`: [Day 0](https://md.sigma2.no/lraXdx4ASqi750xkslK63A), [Day 1](https://md.sigma2.no/teGo9s3VTHC5SOpUfA-luQ), [Day 2](https://md.sigma2.no/rcsdAT6WTgKVG_t3OyOFVg)

### 2021
- {ref}`training-2021-autumn`: [Day 1](https://md.sigma2.no/2LFOgejcSWy7m5soMzk3HQ), [Day 2](https://md.sigma2.no/QCnLXJVgTiqByDWYaA7Gjg), [Day 3](https://md.sigma2.no/11kQBgeET8aqTWSeXu0fNw), [Day 4](https://md.sigma2.no/lO9SXtIOSzuyyaN4elYf8w)
