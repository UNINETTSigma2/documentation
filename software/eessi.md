(eessi)=

# EESSI

**The European Environment for Scientific Software Installation**

EESSI is an innovative service to make optimized scientific software
installations available on any machine anywhere in the world in near real-time -
without the need to build or install the software. They work similarly to
popular streaming services for videos and music.

For the impatient, the single command to get access to EESSI (on NRIS operated
systems) is just

    module load EESSI/2023.06

Thereafter, software can be used via environment modules, for example,

    module load GROMACS/2024.1-foss-2023b

just like it would be used with a traditionally provided installation.

```{note}
When loading the EESSI module and using software provided by
EESSI for the first time, one may experience a delay because data might
need be downloaded from a remote server. EESSI uses [CernVM-FS](https://cernvm.cern.ch/fs/) as
distribution technology. CernVM-FS employs caching at various levels to
provide good startup performance (shown to be equal or better than when
software is hosted on a local parallel filesystem, see
[Performance aspects of CernVM-FS](https://multixscale.github.io/cvmfs-tutorial-hpc-best-practices/performance/)).
```

**Read more about EESSI**
```{toctree}
:maxdepth: 1
eessi/eessi_motivation.md
eessi/eessi_access_on_nris.md
eessi/eessi_using.md
```


(eessi-topics)=
**More information about EESSI:**
  
- Adding software to the shared software stacks provided by EESSI, see [Adding software to EESSI](https://www.eessi.io/docs/adding_software/overview/).
- Building and installing software packages on your local machine (HPC or other)
  on top of EESSI, see [Building software on top of EESSI](https://www.eessi.io/docs/using_eessi/building_on_eessi/).
- Getting access to EESSI on any machine (your own, in the Cloud, in
  CI, ...). see [Getting Access to EESSI](https://www.eessi.io/docs/getting_access/is_eessi_accessible/).
