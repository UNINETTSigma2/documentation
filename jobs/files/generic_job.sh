#!/bin/bash

# Job name:
#SBATCH --job-name=YourJobname
#
# Project:
#SBATCH --account=nnXXXXk
#
# Wall time limit:
#SBATCH --time=DD-HH:MM:SS
#
# Other parameters:
#SBATCH ...

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load SomeProgram/SomeVersion
module list

## Do some work:
YourCommands
