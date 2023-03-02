---
orphan: true
---

(lumi)=

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

| Details | LUMI-G |
|:--------|:-----|
| Peak performance | 375 PetaFLOPs |
| CPU type | AMD® Trento™ 64-Core |
| GPU type | AMD® Instinct™ MI250X GPU |
| GPU memory | 128 GB HBM2e per GPU |
| Node configuration | 1 CPU and 4 x GPUs |
| Number of nodes | 2560 |
| Interconnect | Cray Slingshot 200 Gbit/s, GPUs directly connected to interconnect |
| Storage capacity | <ul><li>117 PB total <ul><li>7 PB fast flash storage</li> <li>80 PB parallel filesystem</li> <li>30 PB object storage</li></ul></li></ul> |
| Expected Top500 | Top 3 |

[Full system specification](https://www.lumi-supercomputer.eu/lumis-full-system-architecture-revealed/)

## LUMI-G
LUMI-G is the main partition of LUMI and is based on AMD accelerators. The main
interactions with the accelerators is through
[`ROCm`](https://rocmdocs.amd.com/en/latest/), `OpenMP` and
[`HIP`](https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html).
CUDA is *not* supported on LUMI and existing users should consider porting their
application to `HIP` through the tools offered. Starting early with the porting
effort is very important and will be supported by the EuroHPC CC team.

Once LUMI-G is operational and pilot testing is completed, all interested users
will be able to request access. Applications that can take advantage of GPU
accelerators will see massive speed-ups on LUMI-G and NRIS will continue to aid
in porting applications to this architecture, see
{ref}`here for more information about our GPU support <extended-support-gpu>`.

### Porting to accelerators
Since LUMI-G is based on AMD GPU Accelerators, not all applications will be able
to instantly take advantage of the additional compute power. AI researchers
using one of the larger frameworks, such as `TensorFlow` and `pyTorch`, will be
able to use LUMI-G directly.

NRIS is still building documentation for taking advantage of
accelerator resources. Researchers that want to begin the transition should
evaluate `OpenACC` (see our {ref}`dev-guides_gpu`),
`OpenMP`
or directly using [accelerated
libraries](https://rocmdocs.amd.com/en/latest/ROCm_Libraries/ROCm_Libraries.html).

If you are interested in porting your application to GPUs, or already have
ported your application and need assistance transitioning to AMD GPUs, 
{ref}`please contact NRIS support <extended-support-gpu>`.

## LUMI-C
LUMI-C is the compute partition of LUMI, dealing with CPU based HPC
applications. Users interested in this partition should also consider the other
clusters already in operation in Norway.
