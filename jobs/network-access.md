

# No network access on compute nodes

According to our security policy, compute nodes shall not have direct access to
the public internet. This means that commands like `git clone` or `conda
install` or `pip install`, updating Git submodules, fetching data from web
sources, etc., will not work on compute nodes.

Compute nodes are the nodes where your jobs which you have submitted via Slurm are running.

You can run commands like `git clone` etc. on the login nodes (the nodes on which you land
when you connect to the computer using ssh).
