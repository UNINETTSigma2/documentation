(eessi-motivation)=

# Motivation for EESSI

Scientific software is a key tool for nearly every research on all kinds of
systems from personal laptops to servers to virtual machines in the cloud and
to supercomputers. Getting scientific software installed and maintained is no
easy task - even for IT professionals with years of experience in building
software.

Over the past decade tools such as [EasyBuild](https://easybuild.io),
[Spack](https://spack.io), [Conda](https://docs.conda.io/en/latest/),
containers, etc. have gained popularity to make the life of software managers
and fearless users easier. Yet, even with such tools managing software remains
a time consuming and tedious task. This is because we witness an explosion of
available software, emerging CPU and GPU architectures, a broader variety of
systems/platforms and more widespread demands to have access to scientific
software quickly, consistently, and everywhere.

*<p style="text-align: center;">What if one wouldn't have to install the software
at all?</p>*

*<p style="text-align: center;">What use cases would be simplified or even made
possible if one can just pick a software from a large collection and start using
it without much delay?</p>*

<p style="text-align: center;">Such services already exist for videos or
music.</p>

*<p style="text-align: center;">Wouldn't it be great if software could be offered
in the same way?</p>*

[EESSI - the European Environment for Scientific Software Installations (EESSI)](https://eessi.io/docs/)
started in 2020 as a collaboration between several Norwegian and
European High-Performance Computing (HPC) sites and industry partners.
EESSI uses the technologies ([CernVM-FS](https://cernvm.cern.ch/fs/),
[Gentoo Prefix](https://wiki.gentoo.org/wiki/Project:Prefix),
[EasyBuild](https://easybuild.io) and [Lmod](https://lmod.readthedocs.io/en/latest/)).

Instead of building or installing software again and again, in EESSI 
software is built ***once*** and distributed via CernVM-FS to any (Linux) computer anywhere
in the world in near real-time. To decouple the software installations from
the Linux distributions being used on a machine (Ubuntu, CentOS, RHEL, Rocky,
...) we use Gentoo Prefix as a compatibility layer. The actual scientific
software is built with EasyBuild. For current and popular CPU architectures
separate installation stacks are pre-built and optimized to not compromise on
performance. Software is made accessible via environment modules using Lmod.

Once a machine is set up to have access to the EESSI software stack,
there is nothing new to learn for a user who is used to running a command such
as

    module load GROMACS/2024.1-foss-2023b

**Read about [getting access to the EESSI stacks](eessi-access-on-nris).**
