

# Interactive jobs

Sometimes you might want to test or debug a calculation interactively but
running interactively on the login node is discouraged and not an option.

What you can do instead is to ask the batch system to allocate a node for you
and once it is assigned to you, you can run interactively on the node for up to 1 hour:

    $ srun --nodes=1 --time=01:00:00 --account=nnXXXXk --qos=devel --pty bash -i


The arguments between `srun` and `--pty` could
be any arguments you would have given to `sbatch` when submitting a
non-interactive job. However `--qos=devel` is probably a good idea to avoid
waiting too long in the queue.
