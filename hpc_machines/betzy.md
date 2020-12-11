# Betzy

Named after [Mary Ann Elizabeth (Betzy) Stephansen](https://en.wikipedia.org/wiki/Elizabeth_Stephansen), the first Norwegian woman to be awarded a doctorate degree.


### The most powerful supercomputer in Norway

Betzy is a BullSequana XH2000, provided by Atos, and will give Norwegian researchers almost three times more capacity than previously, with a theoretical peak performance of 6.2 PetaFlops. The supercomputer is located at NTNU in Trondheim, and was set in production 24 November 2020. Betzy provides close to two billion CPU hours per year, and has a contracted operating time of four years (2024), with an optional one-year extension (2025).


| Details     | Betzy     |
| :------------- | :------------- |
| System     |BullSequana XH2000  |
| Max Floating point performance, double     |	6.2 Petaflops  |
| Number of compute nodes     |	1344  |
| CPU type     |	AMD® Epyc™ 7742 2.25GHz  |
| CPU cores in total  |	172032  |
| CPU cores per node  | 128  |
| Memory in total    |	336 TiB  |
| Memory per node    |  256 GiB  |
| Operating System   | Red Hat Enterprise Linux 7 |
| Total disc capacity     |	2.5 PB  |
| Interconnect  |	InfiniBand HDR 100, Dragonfly+ topology |
| Top500 June 2020 | 55th place \@ 1250 nodes, 76% efficiency|

Almost all components are liquid cooled resulting in a very high cooling efficiency, 95% of heat being captured to water.


### No network access on compute nodes

According to our security policy, compute nodes shall not have direct access to
the public internet. This means that commands like `git clone` or `conda
install` or `pip install`, updating Git submodules, fetching data from web
sources, etc., will not work on compute nodes.
