---
orphan: true
---

(2026-Olivia-OnBoarding)=

# Olivia OnBoarding 2026 seminar series

Drawing on our experience with introducing the Olivia machine and associated services, *NRIS Training* is now offering a series of seminars that provide a deeper introduction to some highly relevant topics specific for the GPU heavy Olivia machine. These seminars are at a basic-to-intermediate level, and targeted towards participants at the preceding OnBoarding event. However, these seminars will also be open to others.

## Practical Information
- Basic command line/linux workflows are expected to be known. (elements of the [HPC Onboarding course given April 14-16.2026](https://documentation.sigma2.no/training/past/2026-04-hpc-on-boarding.html).
- The course is open to all and free of charge. However, **signup is now closed**.
- We will use NIRD, Saga and Olivia for demos and hands-on sessions.
- Collaborative Document for Q&A during the workshop will be used in the same style as for the HPC Onboarding course. Link to this will be provided during the session in question. 
- You will find all relevant documentation linked here at these pages. - This is an online course via zoom. Participants require access to a computer
(not provided by the course organisers) with internet connectivity.

Below you find the list of episodes. Each episode is self-contained and you can join only for the episode(s) you find useful. But not that some content from the previous events may be relevant for the events after - thus the order of things.

<details><summary><H2 style="display:inline">Episode 1: Wed. 22.04.26 (09:00-12:00 CET) - NIRD and File system usage on Olivia</summary><br>

<H3> Content:

- This first episode will cover data staging and data integration between NIRD and Olivia. 
	- We will demonstrate the different options available on Olivia and discuss the best option based on the amount and type of data. 
	- We will also cover using S3 to get data from NIRD Datalake and NIRD Research Data Archive, include some overview of Olivia compared to Saga. 
- In addition, we plan to have give some overview on Cray vs. NRIS Software environments on Olivia.

<H3> Event schedule: 

- 09:00: Start
- 12:00: Finished

<H3> Course material: 

- [NIRD and File system usage on Olivia](https://training.pages.sigma2.no/tutorials/olivia-nird-filesystem-usage/) 
- [Oliva Best Practises Guide](https://documentation.sigma2.no/hpc_machines/olivia.html#olivia-best-practices-guide)

<add link to training material><br>

<H3> Instructors: 

- Siri Kallhovd
- Saerda Halifu
- Ole Widar Saastad


```{note}
- [Video Recording](When done)
- [Q&A](when done))
```

</details><br>

<details><summary><H2 style="display:inline">Episode 2: Wed. 29.04.26 (09:00-12:00 CET) - Containers and software install on Olivia</summary><br>

<H3> Content:

- The second day is concerned with containers and how to utilize them on Olivia. Containers are a key part of the software ecosystem on Olivia.
	- Containers alleviate the stress on the shared file system from Python and R package installations, which normally involve a huge number (tens or hundreds of thousands) of small files.
	- A container is instead a self-contained image that includes all the software and dependencies needed to run an application, which both simplifies software management and improves file system performance on HPC systems.
	- On Olivia, native usage of `pip` and Anaconda or Miniconda is forbidden. Containers are the recommended way to install and use Python and R packages.

<H3> Event schedule: 

- 09:00: Start
- 12:00: Finished

<H3> Course material:

<add link to training material>

- [Containers on Olivia](https://training.pages.sigma2.no/tutorials/olivia-containers/)
- [Oliva Best Practises Guide](https://documentation.sigma2.no/hpc_machines/olivia.html#olivia-best-practices-guide)

<H3> Instructors: 

- Bjørn Lindi
- Morten Ledum
- Jörn Dietze


```{note}
- [Video Recording](When done)
- [Q&A](when done))
```

</details><br>

<details><summary><H2 style="display:inline">Episode 3: Wed. 06.05.26 (09:00-12:00 CET) - GPU utlization and usage on Olivia</summary><br>

<H3> Content: 

- The third and final day will be an introduction to GPU's and how to utilize them on Olivia.

This lecture, titled “Modern Compute Architecture: From CPU to GPU,” provides an introduction to the key concepts behind modern accelerated computing. It begins by exploring the GPU advantage, highlighting why GPUs have become essential for high-performance computing, AI, and scientific applications. 

The session then presents a CPU and GPU architecture overview, explaining the fundamental differences between traditional processors and massively parallel accelerators.
Building on this foundation, the session examines the architecture of modern GPUs, with a comparative look at NVIDIA and AMD designs, focusing on the architectural features that drive performance. 

Finally, the contribution introduces the GPU software ecosystem, presenting the roles of CUDA and ROCm as the main programming platforms for GPU-accelerated applications.

<H3>Course material: 

<add link to training material>

<H3> Instructors: 

- Hicham Agueny
- Binod Baniya
- Magnar Bjørgve


```{note}
- [Video Recording](When done)
- [Q&A](when done))
```

</details><br>


```{note}

We advise you to apply for access to Olivia and the training accounts NN9970K and NS9970K to attend these seminars.

Follow the link below to apply for a user account on the national HPC resources:

**Apply for access [here](https://www.metacenter.no/user/application/form/hpc/)**

* For the project, please select **NN9970K**: Training and Outreach. 
* Under resource, choose **Olivia** and **Saga**
* Set the account start date as the date of application
* Set the end date to **2026-05-30**.
* The application procedure is documented [here](https://www.sigma2.no/how-apply-user-account).

We will assume that all of you who apply for access to NN9970K also want access to NS9970K unless exlicitely notified us. Please also note that it takes about one day for everything to be settled, so please **RESPECT THE APPLICATION DEADLINE**.

 

*(This information is duplicated in the signup form.)* 

```

### Coordinators

- Espen Tangen
- Eirik Skjerve

### Code of Conduct

All course participants are expected to show respect and courtesy to
others. We follow the [carpentry code of
conduct](https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html#code-of-conduct-detailed-view).
If you believe someone is violating the Code of Conduct, we ask that you report
it to [the training team](mailto:training@nris.no).

### Contact us

You can always contact us by sending an email to [support@nris.no](mailto:support@nris.no).





