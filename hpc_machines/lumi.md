# LUMI

> The European High-Performance Computing Joint Undertaking (EuroHPC JU) is
> pooling European resources to develop top-of-the-range exascale supercomputers
> for processing big data, based on competitive European technology.
>
> At the time of installation, LUMI will be one of the world’s fastest computer
> systems, having theoretical computing power of more than 500 petaflops which
> means 500 quintillion calculations per second. LUMI’s performance will be more
> than tenfold compared to one of Europe’s fastest supercomputer today (Piz
> Daint, Switzerland). LUMI will also be one of the world’s leading platforms
> for artificial intelligence.

```{note}
LUMI will be installed in Kajaani, Finland. Basic operations are done by the
vendor and CSC. Researchers will be supported by different teams: the Norwegian
EuroHPC competence centre for porting and tuning applications on the LUMI
hardware and programming environment, Sigma2 for handling allocation requests,
and The LUMI User Support Team (LUST) for day-to-day issues.
```

## Overview
LUMI is the first of a new class of pre-exascale supercomputers set up by [a
consortium of countries in
Europe](https://www.lumi-supercomputer.eu/lumi-consortium/). All partner
countries will get access to an equal share of the resources, currently
estimated to be `2%` for Norway. In addition, project can apply for resources
through EuroHPC JU which control the other `50%` of LUMI capacity.

LUMI is aimed at AI and HPC workloads that can take advantage of GPU
accelerators.

LUMI is divided into the following partitions, where the largest will be LUMI-G
followed by LUMI-C:
![LUMI partition
overview](https://www.lumi-supercomputer.eu/content/uploads/2020/11/lumiSlide-1024x576.png)

| Details | LUMI |
|:--------|:-----|
| Peak performance | 552 PetaFLOPs |
| CPU type | Next generation AMD® Epyc™ 64-Core |
| GPU type | Next generation AMD® Instinct™ GPU |
| Storage capacity | <ul><li>117 PB total <ul><li>7 PB fast flash storage</li> <li>80 PB parallel filesystem</li> <li>30 PB object storage</li></ul></li></ul> |
| Expected Top500 | Top 3 |

## LUMI-G
LUMI-G is the main partition of LUMI and is based on AMD accelerators. The main
interactions with the accelerators is through
[`ROCm`](https://rocmdocs.amd.com/en/latest/), `OpenMP` and
[`HIP`](https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html).
CUDA is *not* supported on LUMI and existing users should consider porting their
application to `HIP` through the tools offered. Starting early with the porting
effort is very important and will be supported by the EuroHPC CC team.

The Metacenter is currently soliciting pilot users for LUMI-G participation.
Interested users should contact
[support@metacenter.no](mailto:support@metacenter.no). Since there is a limited
number of pilots, users will need to dedicate some amount of time to get ready
for pilot testing, however, the Metacenter will be available for help in porting
applications to LUMI-G.

Once LUMI-G is operational and pilot testing is completed, all interested users
will be able to request access. Applications that can take advantage of GPU
accelerators will see massive speed-ups on LUMI-G and the Metacenter will
continue to aid in porting applications to this architecture.

### Porting to accelerators
Since LUMI-G is based on AMD GPU Accelerators, not all applications will be able
to instantly take advantage of the additional compute power. AI researchers
using one of the larger frameworks, such as `TensorFlow` and `pyTorch`, will be
able to use LUMI-G directly.

The Metacenter is still building documentation for taking advantage of
accelerator resources. Researchers that want to begin the transition should
evaluate `OpenACC` (our documentation can be found
[here](https://documentation.sigma2.no/code_development/guides.html)), `OpenMP`
or directly using [accelerated
libraries](https://rocmdocs.amd.com/en/latest/ROCm_Libraries/ROCm_Libraries.html).

## LUMI-C
LUMI-C is the compute partition of LUMI, dealing with CPU based HPC
applications. Users interested in this partition should also consider the other
clusters already in operation in Norway.
