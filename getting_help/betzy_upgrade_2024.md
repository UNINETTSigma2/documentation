---
orphan: true
---

(betzy-upgrade-2024)=

# Major Upgrade of Betzy 2024

During the summer of 2024, we will upgrade our supercomputer Betzy in two
phases. The first phase involves only the administration and backed nodes and
shouldn't affect users except for the downtime in June/July.

During the second phase, we will upgrade the operating system and software of
all login and compute nodes. This process requires the recompilation and
installation of all scientific software and libraries, including compiler tool
chains.

## Tool Chains

Maintaining old software consumes a significant amount of time, and as we are a
small team, we must focus our resources on maintainable software packages and
versions to provide good support for our users. Therefore, we will only install
tool chains and software packages that are at least somewhat up-to-date. We
have decided to install the last four published versions of the FOSS (based on
GCC) and Intel tool chains (as distributed by EasyBuild).

The following tool chains will be available after the update:
- `foss/2022a`
- `foss/2022b`
- `foss/2023a`
- `foss/2023b`
- `intel/2022a`
- `intel/2022b`
- `intel/2023a`
- `intel/2023b`

Additionally, the following derivatives will also be available after the upgrade:
- `gompi/2022a`
- `gompi/2022b`
- `gompi/2023a`
- `gompi/2023b`
- `iompi/2022a`
- `iompi/2022b`
- `iompi/2023a`
- `iompi/2023b`

### Tool Chains That Will Be Removed

The following tool chains will be removed during the upgrade and will **not**
be available any longer:
- `foss/2019a`
- `foss/2019b`
- `foss/2020a`
- `foss/2020b`
- `foss/2021a`
- `foss/2021b`
- `intel/2019a`
- `intel/2019b`
- `intel/2020a`
- `intel/2020b`
- `intel/2021a`
- `intel/2021b`

The same is true for derivative tool chains like gompi or iompi based on these versions.

## Scientific software

Almost all modules installed on Betzy are based on one of the above-mentioned tool chains. We will reinstall newer versions of software packages based on the up-to-date tool chains. In most cases, that means that you will get a newer, more recent version.

Only very few packages will not be reinstalled and will not be available after the upgrade, see the list below.

You can check which easyconfigs, the recipes we use for installing, are available here:
<https://github.com/easybuilders/EasyBuild-easyconfigs/tree/develop/EasyBuild/easyconfigs/>

If you critically depend on a certain version, please contact us at [contact@sigma2.no](mailto:contact@sigma2.no), and we will explore all options.

### Software packages that will be removed

The following packages are only available for older tool chains and will therefore be removed during the upgrade:
- `Globus-CLI/3.6.0-GCCcore-11.2.0`
- `GRASP/2018-foss-2019b`
- `Keras/2.4.3-fosscuda-2020b`
- `PCMSolver/1.3.0-gompi-2020b`
- `Theano/1.1.2-fosscuda-2020b-PyMC`
- hipSYCL: `hipSYCL/0.9.1-gcccuda-2020b` and `
  hipSYCL/0.9.2-GCC-11.2.0-CUDA-11.4.1`

If your work depends on one of those packages, please write to us at [support@nris.no](mailto:support@nris.no), and we will find a solution together.
