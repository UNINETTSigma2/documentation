(castep)=

# CASTEP

```{note}
If you want to contribute to the documentation of this code, please contact us at [support@nris.no](mailto:support@nris.no)
```

```{contents} Table of Contents
```
## Introduction

CASTEP is a leading code for calculating the properties of materials from first principles. Using density functional theory, it can simulate a wide range of properties of materials proprieties including energetics, structure at the atomic level, vibrational properties, electronic response properties etc. In particular it has a wide range of spectroscopic features that link directly to experiment, such as infra-red and Raman spectroscopies, NMR, and core level spectra.

More information on the [CASTEP
website](https://www.castep.org).


(access)=
### License and access

The CASTEP Developers' Group (CDG) and Cambridge Enterprise have announced a
cost-free worldwide source code license to CASTEP and NMR CASTEP for academic
use.

To get access to CASTEP, you need to follow the procedure described below. 

If you, however, wonder if you are in the castep group of users, you can find out by typing: 

```bash
id | tr "," "\n" | grep castep
```
If this command comes out with a result, then you are in the group and may use the code - if not, go through the following procedure:

```
Access to CASTEP is limited by membership in the castep group.
In order to use the software on our infrastructure, first make sure you have access to a valid license.

**When you have access to a valid CASTEP license**
Send an email to
<a href="mailto:contact@sigma2.no?subject=Request for CASTEP access">
contact@sigma2.no</a> with the following information:
* Full name
* E-mail address
* ORCA ID

We will then have to manually verify this information with STFC UK/CASTEP before granting access. As such there may be some waiting time unfortunately.
```

### Citation

For the recommended citation, please consult the [CASTEP page](https://www.castep.org).

## CASTEP on NRIS machinery

Currently, **CASTEP** is installed on {ref}`saga`. To see available versions when logged on to the machine in question, use the ```module avail``` or ```module spider``` commands as shown below:

```bash
module avail castep
```
If you are in the castep group of users, you may use CASTEP by typing:

```bash
module load CASTEP/<version>
# (eg. module load CASTEP/22.1.1-intel-2022b)
```
specifying one of the available versions.

### Running CASTEP on NRIS machines

**Please inspect the job script examples and/or jobscripts before submitting jobs!**

<details>
<summary>Testing CASTEP: Ethene
</summary>
<br>
To test CASTEP, we have borrowed the Ethene-example from [www.mjr19.org.uk/castep/test.html](https://www.mjr19.org.uk/castep/test.html). To perform this test, you need two files; 

One file called **ethene.cell** with the contents

```{literalinclude} ethene.cell
:language: bash
```
and one called ethene.param with the contents

```{literalinclude} ethene.param
:language: bash
```

Running **CASTEP** would produce an ethene.castep file (amongst others) within seconds for the running examples provided below. Towards the end of this file, there are final structure energy outputs printed, ```Final energy, E```; values here should be in the range of -378.015eV.

A subset of the benchmark sets, the medium set al3x3 and solid benzene together with the ethene-example used here has been added to the CASTEP home folder on both Saga. You get them into your working directory by typing

```bash
cp /cluster/software/CASTEP/benchmarks/* .
```

</details>

<details>
<summary>Running CASTEP on Saga 
</summary>
<br>
On Saga, the nodes have more memory per core than on Betzy and you are allowed to share nodes with others. Thus the specs on memory in runscript example below. Note that, due to the higher memory amount you may be able to use more cores/node on Saga. Note, however, that there is a 256 core limit on Saga. 
 
```{literalinclude} saga_castep.sh
:language: bash
```
</details>

