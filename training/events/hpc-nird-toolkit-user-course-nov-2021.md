# HPC and NIRD toolkit user course November 2021

In November 2021, the Norwegian Research Infrastructure Services **NRIS**
(formerly known as the Metacenter) partners Sigma2, UiB, UiO & UiT  are jointly
offering a hands-on course for current and future users of national HPC systems
(Saga, Fram, Betzy) and the NIRD Toolkit. The course is open to all users and
free of charge.

For individual sessions we list recommended prerequisites. The participants can
choose which session they want to attend and you're certainly welcome to just
attend all sessions you are interested in. On each course day, we will start at
09:00 and conclude at around 13:00. Each "morning" will include presentations
and hands-on time on one of the actual HPC systems or the NIRD Toolkit. We
recommend that participants use the afternoon on each day to repeat examples
from the morning to get more experience on the systems.


## **Practical Information**

This is an online course via zoom. Participants require access to a computer
(not provided by the course organisers) with internet connectivity and
pre-installed programs to participate in the video meeting of the course (zoom,
might work from a browser), to access the HPC systems (e.g., ssh or Putty), to
transfer data to/from an HPC system (e.g., scp or WinSCP) and to access the
NIRD Toolkit (web browser).

```{note}

1. Those who do not have an account on one of the HPC machines or NIRD
   (storage, Toolkit, ...) yet need to apply for an account. The application
   page is available
   [here](https://www.metacenter.no/user/application/form/notur/) and
   documentation for the application procedure is available
   [here](https://www.sigma2.no/how-apply-user-account). You are applying for
   an HPC account. For the project, please select “NN9987K: Course Resources as
   a Service #3” (the Project manager will be filled in automatically), choose
   Saga under Resources, set the Account start date to 2021-10-26 and the
   Account end date to one week after the course, that is 2021-11-12.

2. In order to use the NIRD Toolkit during the course you will need either a
   Feide account or a Feide OpenIDP guest account.  You can create a Feide
   OpenIDP guest account at https://openidp.feide.no/.  See your
   eduPersonPrincipalName (eduPPN) at http://innsyn.feide.no . Those who  are 
   using Feide openidp guest account should go to minside.dataporten.no to find    eduppn, and it is the field called "Secondary user identifiers".
```

You can always contact us by sending an email to [support@nris.no](mailto:support@nris.no).

If you want to receive further information about training events, and other announcements about IT resources and services for researchers, there are a couple of information channels you can subscribe to:
- users at University of Bergen and affiliated institutes: [register to hpcnews@uib.no](https://mailman.uib.no/listinfo/hpcnews)
- users of Sigma2 services: [subscribe to Sigma2's newsletter](https://sigma2.us13.list-manage.com/subscribe?u=4fd109ad79a5dca6dde7e4997&id=59b164c7b6)

**Dates: 2021 November 2, 3, 4 and 5 
Time: each day 09:00-13:00 with short breaks**
Location: Video meeting (zoom), we use HackMD for asking questions (information will be send to registered participants via email). **Note, sessions will be recorded and made publicly available after the course!**

Participants may choose the lectures which they are interested in.


## **Registration**

The course is free of charge and is organised by the NRIS partners Sigma2, UiB, UiO & UiT.
**Registration deadline for the course is October 20th.**
**Register your participation [here](https://skjemaker.app.uib.no/view.php?id=10882801)**


## Code of Conduct

All participants in our course are expected to show respect and courtesy to
others. We follow [carpentry code of
conduct](https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html#code-of-conduct-detailed-view).
If you believe someone is violating the Code of Conduct, we ask that you report
it to [the training team](mailto:training@nris.no).


## **Agenda**


### Day 1: 02-11.2021: Welcome, Introduction to High Performance Computing

- **Get ready: 08:50-09:00 Connect to zoom meeting**

**Prerequisites:** Command line experience is necessary for this lesson. We
recommend the participants to go through
[shell-novice](https://swcarpentry.github.io/shell-novice/), if new to the
command line (also known as terminal or shell)

**Course material [here](https://sabryr.github.io/hpc-intro/)**

- **Session 1: 09:00-09:55 Welcome, What is HPC, working on HPC System, Scheduling Jobs**

Break for 10 minutes

- **Session 2: 10:05-11:00 Accessing Software, transferring files**

Break for 10 minutes

- **Session 3: 11:10-12:05 Running a parallel job, Using resources effectively**

Break for 10 minutes

- **Session 4: 12:15-12:45 HPC through a web browser**

- **Session Q&A: 12:45-13:00 Questions & Answers**


### Day 2: 03-11-2021: Optimization of codes and Resource Management

**Prerequisites:** Basic HPC knowledge, we recommend participants  to attend Day1 sessions

- **Get ready: 08:50-09:00 Connect to zoom meeting**

- **Session 1: 09:00-09:45 Optimization of codes and batch script (part I)**

Break for 10 minutes

- **Session 2: 09:55-10:30 Optimization of codes and batch script (part II)**

Break for 10 minutes

- **Session 3: 10:40-11:20 How to choose memory and number of cores (part I)** (R. Bast)
  - [Slides](https://cicero.xyz/v3/remark/0.14.0/github.com/bast/talk-better-resource-usage/main/talk.md/)
  - [Tutorial about memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)
  - [Tutorial about cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)

Break for 10 minutes

- **Session 4: 11:30-12:10 How to choose memory and number of cores (part II)** (R. Bast)
  - [Slides](https://cicero.xyz/v3/remark/0.14.0/github.com/bast/talk-better-resource-usage/main/talk.md/)
  - [Tutorial about memory](https://documentation.sigma2.no/jobs/choosing-memory-settings.html)
  - [Tutorial about cores](https://documentation.sigma2.no/jobs/choosing-number-of-cores.html)

Break for 10 minutes

- **Session 5: 12:20-12:50 Optimization of codes: Usecase**
- **Session Q&A: 12:50-13:00 Questions & Answers**


### Day 3: 04-11-2021:  Containers on HPC and NIRD Toolkit

- **Get ready: 08:50-09:00 Connect to zoom meeting**

- **Session 1: 09:00-10:20 Containers on HPC (part I)**

Break for 10 minutes

- **Session 2: 10:35-11:30 NIRD Toolkit (part I)**

Break for 10 minutes

- **Session 3: 11:40-12:40 NIRD Toolkit (part II)**

Break for 10 minutes

- **Session Q&A: 12:40-13:00 Questions & Answers**


### Day 4: 05-11-2021: GPU Computing

- **Get ready: 08:50-09:00 Connect to zoom meeting**

- **Session 1: 09:00-09:45 Introduction to GPU computing** (J. Nordmoen)
  - [Slides can be found here](https://docs.google.com/presentation/d/e/2PACX-1vSz2-a0FzWkMgSICvQpDAXcCgVcyMryimjlZGI_DOyDaT6iWJ5ZOb7WGpNZ9FT6ZpXPWyP1nDxWXLLr/pub?start=false&loop=false&delayms=3000)

Break for 15 minutes

- **Session 2: 10:00-10:45 Using GPUs at NRIS** (J. Nordmoen)
  - [Slides can be found here](https://docs.google.com/presentation/d/e/2PACX-1vSz2-a0FzWkMgSICvQpDAXcCgVcyMryimjlZGI_DOyDaT6iWJ5ZOb7WGpNZ9FT6ZpXPWyP1nDxWXLLr/pub?start=false&loop=false&delayms=3000)

Break for 15 minutes

- **Session 3: 11:00-11:45 Practical introduction to using GPUs** (S.R. Jensen)

Break for 15 minutes

- **Session 4: 12:00-12:45 Containers on HPC (part II): MPI and GPU** (S.R. Jensen)

- **Session Q&A: 12:45-13:00 Questions & Answers**


## Preparing your machine for the course

We assume you have the necessary tools installed on your machine and are able
to use them. You need tools to login into a remote machine (e.g., ssh or PuTTY)
and to transfer data to/from a remote machine (e.g., scp or WinSCP). If you
need to install such tools, please see [prepare your machine
section](https://wiki.uib.no/hpcdoc/index.php/HPC_and_NIRD_toolkit_course_fall_2020#Preparing_your_machine_for_the_course).


### Previous Course

Our previous course page can be found
[here](https://wiki.uib.no/hpcdoc/index.php/HPC_and_NIRD_toolkit_user_course_March_2021)
and video recording of the course will be updated soon.
