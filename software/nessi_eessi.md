(nessi-eessi)=

# NESSI and EESSI

**The Norwegian Environment for Scientific Software Installation**

_and_

**The European Environment for Scientific Software Installation**

NESSI and EESSI are innovative services to make optimized scientific software
installations available on any machine anywhere in the world in near real-time -
without the need to build or install the software. They work similarly to
popular streaming services for videos and music.

For the impatient, the single command to get access to NESSI (on NRIS operated
systems) is just

    module load NESSI/2023.06

and for EESSI it is similarly

    module load EESSI/2023.06

Thereafter, software can be used via environment modules, for example,

    module load GROMACS/2023.1-foss-2022a

just like it would be used with a traditionally provided installation.

```{note}
When loading the NESSI/EESSI modules and using software provided by
NESSI/EESSI for the first time, one may experience a delay because data might
need be downloaded from a remote server. NESSI and EESSI use [CernVM-FS](https://cernvm.cern.ch/fs/) as
distribution technology. CernVM-FS employs caching at various levels to
provide good startup performance (shown to be equal or better than when
software is hosted on a local parallel filesystem, see
[Performance aspects of CernVM-FS](https://multixscale.github.io/cvmfs-tutorial-hpc-best-practices/performance/)).
```

**Read more about the NESSI and EESSI**
```{toctree}
:maxdepth: 1
nessi_eessi/nessi_eessi_motivation.md
nessi_eessi/nessi_eessi_access_on_nris.md
nessi_eessi/nessi_eessi_using.md
```

**Getting support for NESSI and EESSI**

- For getting support for NESSI, please contact us via the standard NRIS support
([support@nris.no](mailto:support@nris.no)).
- For getting support for EESSI, please see [https://www.eessi.io/docs/support/](https://www.eessi.io/docs/support/)

(nessi-eessi-future-topics)=
**In the future, we will add more information about:**

- Adding software installations to the shared software stacks provided by NESSI.
  For EESSI, see [Adding software to EESSI](https://www.eessi.io/docs/adding_software/overview/).
- Building and installing software packages on your local machine (HPC or other)
  on top of NESSI. For EESSI, see [Building software on top of EESSI](https://www.eessi.io/docs/using_eessi/building_on_eessi/).
- Getting access to NESSI on any machine (your own, in the Cloud, in
  CI, ...). For EESSI, see [Getting Access to EESSI](https://www.eessi.io/docs/getting_access/is_eessi_accessible/).
