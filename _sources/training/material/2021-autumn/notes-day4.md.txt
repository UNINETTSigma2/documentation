---
orphan: true
---

(training-2021-autumn-notes-day4)=
# Questions, answers and notes from day 4

## Ice breaker

(Add an `o` after the option `|` to "vote" for it, vote for as many as you like)

- Have you heard about the following GPU technologies:
    - CUDA              | oooooo
    - HIP               | o
    - SYCL              |
    - OpenACC           | o
    - OpenMP Offloading | o
    - Kokkos/RAJA       |
- My interest in GPU offloading is:
    - Just curious to hear more about GPUs  | ooooooooooo
    - I want to port my application to GPUs | ooo
    - My application is already ported      |
    - I am new to HPC, I want to broaden my perspective and general knowledge. | ooooo


## Day 4 Q&A

GPUs are excellent for well structured problems.

- As we know, one thread work on cpu, thread group works on GPU. However, can they use different initial data at begin in the one GPU group?
    - Concurrent kernel execution makes GPUs appear more like MIMD (Multiple instructions and Multiple data). So, yes it is possible to start with different initial data but the programmer needs to write his/her program to focus more on multiple data needs. I need to mention this as well that newer Gpus can run kernels simultaneously (they hack into multiple streams and run different kernels parallelly); however, on older Gpus it is not possible and the second kernel runs after the first kernel, somewhat serially. It is recommended to perform SIMD operations on one GPU. If the tasks are not independent then the inter-communication within MIMD may drop down the over-all performance.
    - Yes, each thread in the thread group can use different memory locations when calculating, so they can load different data, but have to perform the same calculations

- Does OpenACC support conditional execution that blocks instructions of individual threads? To my understanding, this is supported by CUDA. 
    - In OpenACC you need to add a reduction clause yourself, and the compiler will generate a code to perform if-condition.-
    - There is no explicit support for calculating your thread ID within OpenACC, however, that does not mean that you cannot get thread divergence with OpenACC
```c
// The following is likely to result in thread divergence and very low performance on a GPU
#pragma acc kernels
for (int i = 0; i < X; i++) {
    if (i < Z) {
        i -= 1;
    } else {
        i += 1;
    }
}
```

- Are there any parallell distributed blas/lapack libraries for GPUs, or do you know of any initiatives in this direction (thinking PBLAS/ScaLapack, or similar libraries with GPU support)?
    - GPU vendors have their own library for Blas, for example, nVIDA has CUBLAS to support Blas operations on nVIDIA Gpu architecture.
    - See the slides for links to the most popular vendor's BLAS libraries

- Yes, but are there any parallell distributed libaries (or work in this direction), ie. with the memory and operations distributed in parallell? PS: to the best of my knowledge the existing libraries are for single GPU only?
    - cuBLAS, hipBLAS and oneMKL support multiple GPUs on one node (not distributed across nodes)
    - Some hybrid library may achieve the desired thing, for example, StarPU library which being developed at Inria France.

- Can you give some examples of scientific applications that benefit from running on GPUs?
    - e.g. testing the limitation of quantum chemistry package, which is one of the project that will be run on LUMI. 
    - Examples in chemistry:
        - TeraChem (PetaChem)
        - Gromacs (LUMI pilot)
        - GAMESS
        - BigDFT
        - NWChem(Ex)
        - VeloxChem (LUMI pilot)
        - Bifrost (Stellar atmosphere code, very large  application, largest user of core hours on the Sigma2 systems)
    - What about Gaussian?
        - Yes, and Gaussian :)
            - I love you!

- Examples of GPU usage in genomics research?
    - And in proteomics?
    - In variant calling you have [DeepTrio](https://github.com/google/deepvariant/tree/r1.2/deeptrio) and [Deepvariant](https://github.com/google/deepvariant)
    - And [here's](https://github.com/BIRL/Perceptron) a random proteomics tool for GPUs I found

- If some code scales up to say 512 CPU cores, is there any hope to make it run more efficiently on a large GPU? We can assume no I/O.
  - There is no direct answer to this question. But it depends on what the code does. If there is more reading files then there will be no optimal gain. BUT in general, you would expect gain in speed when running on GPU by a factor of 5 or even more, and that depends also on the approach you will implement on your code, e.g. openacc, cuda, openMP offloading, HIP...and some experiments have shown that the choice of the compiler helps to get extra performance. 
  - Or to rephrase the question: in which situations can 512 GPU cores be faster than 512 CPU cores?
    - maybe if the execution is done on GPU and there is no transfer from CPU to GPU and back, as this process takes time. GPUs basically have thousands of cores.
          - (Yes, but many problems cannot be partionned in thousands of pieces! Then it would usually scale to more CPU too) 

- For large problems (e.g., high resolution ocean model): is direct GPU-GPU communication supported on SAGA/BETZY/LUMI, or is the data exchange always via the CPU?
  - In general, the data get copied from the cpu to gpu (and back), but then you could run the code also on multiple-GPUs.
  - GPU-to-GPU communication should be supported, and I believe Jørgen is working on an example/tutorial on this for our documentation pages.
  - Apparently only Betzy :)
  - LUMI we will know later ;)
      - LUMI supports direct GPU-GPU communication both intra- and inter-node!

- Are there no GPU nodes on Fram?
  - There is no GPU on Fram.

- Let's say I want to see if I am using GPU cores in my Gaussian calculations, how can I do this, bylooking at script? and if I am not using GPU, how can I start using it so my jobs run faster?
   - Well, you need to inculde the library of the tool you are using, and the syntax that the compiler recongnises. So, the compiler will provide some information illustrating what it does. and this this the case for instance when implementing OpenACC (use openacc as the head in a fortran program).
   - For NVIDIA GPUs, you can ssh into the compute node and run `nvidia-smi` to check what runs on a GPU, how much memory is used and how high its utilisation is.
   - We have on our agenda to document Gaussian with GPUs, but it is a bit difficult since Gaussian has such a strict license
 
 - What is a tool chain?
    - A toolchain handles build and installation processes. Detailed explanation is [here](https://docs.easybuild.io/en/latest/Concepts_and_Terminology.html#toolchains)
    - A toolchain like `foss`, `intel`, `iompi`, `fosscuda` contains many useful tools and libraries for code development, like compiler, MPI library, math libraries, CUDA library etc 

- On Betzy: does it make sense to request less than all CPU/memory resources available of an accel node (i.e. the entire node resources)? Or, are there multiple GPUs associated to the same accel node on Betzy? My reasoning is that if one user uses the entire GPU part of an accel node then there is little point for other users to request CPU resources on that node.
     - ON Betzy there is 4 GPUs per node (with a total of 16 GPUs). and so you dont need to allocate all the GPUs in your slurm script. Maybe your code needs only one GPU or 2 GPUs might be enough. If more users use GPUs then the queue time will be longer. and if there is more demand then more GPUs can be incorporated in the future.
     - Actually it would be nice to other users if you only request about the same fraction of memory and number of CPUs as you do for the number of GPUs.

- Do you need the array slicing if all of the arrays `a`, `b`, and `c` are to be used? In the `#pragma omp ... map()`.
    - I'll try to answer my own question. In C the array lengths are not contained in the arrays themselves, so we need to supply the length?
        - That is the correct assumption, this is actually not needed when using OpenMP offloading within Fortran
    - Could you reformulate your question?
        - In Jørgens OpenMP C code, in the `map()` inside the `pragma`, he specifies `a[0:N]`. My question was why he specified the slice `0:N` if he were to use the entire array, but then I realized this is not Python so the array lengths are not contained in `a`.
            - In deed, the meaning of the syntax is actually `a[array offset, number of elements to copy from offset]`

- Why have they chosen to use AMD GPUs at LUMI when Nvidia and CUDA are the most supported and used in ML / AI?
  - One of the benefit of AMD GPU is the ability to use open source softwares in terms of compilers and so..and I also think that the cost of AMD GPU could be cheaper than NVIDIA GPU.
  - Another possibility is that it was chosen for "politics" to avoid vendor lock-in
  - Lastly, it should be mentioned that the MI200 GPUs that will appear in LUMI is expected to be the fastest most advanced GPUs available so for code that can utilize AMD hardware it will be incredibly powerful
      - Also note that a lot of software is transitioning to support AMD, so Gromacs support LUMI already, Tensorflow as well as PyTorch will also run on LUMI
          - Thats good news for us open source supporters :slightly_smiling_face: 

- What is `NCCL`?
    - `NCCL` stands for [Nvidia Collective Communication Library](https://developer.nvidia.com/nccl) and helps us perform transfers between GPUs as well as collective operations between GPUs (both inter- and intra-node)

- If one wants to program similarly to that of Stig just now, but for an AMD GPU, what tools to use?
    - If you want to program at the same low level as CUDA on AMD hardware [HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html) is the right tool
    - However, depending on your needs, OpenMP offloading is also supported on some AMD hardware, as well as OpenCL.
    - You could also porting OpenACC code to OpenMP to be run on AMD-GPU.

- The module system you use on all nodes, is that possible to install on, say, a regular computer with Ubuntu?
    - Yes, I think [this is the best introduction](https://docs.easybuild.io/en/latest/Installation.html) to installing the module and software tools we have on our clusters
    - I would also recommend to install `Lmod` as the module tool together with EasyBuild
        - Awesome :slightly_smiling_face: It can be a royal pain in the ass to get TensorFlow working with all the correct versions of CUDA and whatnot, so a modular system would be very helpful with this.
            - Tensorflow is always a pain in the behind, but if this is your main code, I can recommend virtual environments (as they are much easier to get started with compared to the modules and software we use) or alternatively... containers ;)
                - Cool! Any pointers (names, websites) to start learning proper virtual environments for easing the use of TensorFlow? I'm only used to miniconda.
                    - [This is a good primer on what virtual environment is and why you should use it](https://realpython.com/python-virtual-environments-a-primer/)
                    - [Here is how Tensorflow recommends installing with `virtualenv`](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)
                        - Thanks!


- general question: Stig just said "the run time varied a lot among repetitions of the same job". I've noticed this as well. Really wilde changes of 300-500%. Any ideas on what this id due to?
    - Difficult to say, but on general terms, we observe filesystem hiccups from time to time which can affect your jobs, the processors should almost always behave in the same way, but the shared filesystem can affect jobs in this way
        - how to deal with this when benchmarking job configurations? Effectively, I ended up with lots of noise.
            - Again, very good questions, ideally there should not be this much variability in the filesystem, but here we are
            - I can recommend to see [here](https://documentation.sigma2.no/files_storage/performance.html) for some tuning options to optimize for the shared filesystem
            - Other recommendations would be to try to batch writes and use [`mmap`-ed files if possible](https://en.wikipedia.org/wiki/Mmap)
                - I guess this implies compiling myself. Or is it something I can do on software that is installed by others(you/admin)?
                    - For "external" application you are restricted to the filesystem options described [here](https://documentation.sigma2.no/files_storage/performance.html) as well as trying to limit the amount of output (maybe checkpoint less often or similar strategies)
         
- What are your thoughts on the new Macs which use shared RAM and VRAM? Does it solve any current problems in, say, machine learning?
    - The biggest advantage of the new ARM processors in these Macs is the increased bandwidth between system memory and the CPU
    - You will most likely not notice that it is shared (which is something Intel and AMD has done for years as well)
        - Are you referring to AMD's Smart Access Memory? Is this technology used in HPC? If i remember correctly, you need both AMD GPU and CPU for this to work.
            - No, I meant shared VRAM has been used by CPUs with integrated graphics for quite some time, both by Intel and by AMD
                - But integrated graphics isnt really usable for any heavy computation? Do we call the GPUs in the new Macs integrated GPU?
                    - The GPU in the new Macs are really powerful, but yes they are integrated, in the sense that they are a part of the SOC, you can't remove the GPU without damaging the CPU in a sense
                        - Has the world seen powerful GPUs using shared memory before? Powerful in the sense that you get a benefit of running for example ML on the GPU instead of the CPU.
                            - The SOC in the Playstation 5 and Xbox Series X are using the same setup with a powerful GPU on the same SOC with shared memory
    - However, the new ARM processors actually have a separate accelerator for ML, but it has to be used with [Apple's proprietary interface](https://developer.apple.com/machine-learning/core-ml/)
        - Tensorflow is coming with support for this accelerator



- Uhm... forgive my ignorance. Wasn't the point of running singularity to pack everithing in a single box to allow reproducibility on different machines? Now it looks like there is substantial linking to external libraries that are machine specific... I'm confused...
    - It is mainly MPI that requires some support from outside the container, other than that your impression is correct

- I love the name of LUMI (snow in Finnish) :)
    - Indeed very nice!
        - thanks, now I know my first word in Finnish!

- Is AMD's Smart Access Memory something that can be / is used in HPC?
    - We expect so, yes
    - Right now we do not have any AMD systems, but we are working on adding this support for our Nvidia systems
        - Where it is called Nvidia BAR
            - Cool! Looking forward to hearing more about it!


## Feedback for the day
- One thing you particularly enjoyed
    - The introduction to GPU computing was great! The other lectures are good too, but (understandably) for more experienced users.
    - I liked the detailed explanation of the parts of the codes.
    - intoriduction to GPU was good
    - I liked the 45+15 min. presentation+break schedule.
- One thing we should remove/change/improve
    - If we know in advance we'll work with BigDFT in the demo, we could download it in advance (took many minutes) and type along with the demo.
    - Stop for questions during the sessions. Many people need the breaks as such, not as extensions of the lecture. One of the assisting teachers can be responsible for bringing up something that is important or of common interest from the HackMD (even if the question has already been answered in written there).
