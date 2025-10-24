---
orphan: true
---


(fram_decom)=

# Fram decommissioning information

**Published:** 2025-10-23

The Fram supercomputer is scheduled for decommissioning on **4 December 2025** as part of the national e-infrastructure upgrade.

All active projects with a quota *only* on Fram, and users are being migrated to the new national supercomputer, **Olivia**. This guide provides the official timeline, data migration plan, and the actions you need to take to ensure your research continues with minimal disruption.

---

## Critical Timeline

| Date/Time | Event | Impact on Your Work |
| :--- | :--- | :--- |
| **Friday, 21 Nov 2025** | **Fram Data Freeze** | `/cluster/project` and `$HOME` directories will become **read-only**. You can only write data to the `/cluster/work` area. |
| **Thursday, 4 Dec 2025**<br>(11:00 CEST) | **System Shutdown** | All compute nodes will be reserved to finish running jobs. Fram will then be powered down and taken offline. |

**Note:** Short-notice downtimes may be necessary. Please monitor [opslog.sigma2.no](https://opslog.sigma2.no) for service announcements.

---

## Data Migration

To facilitate the transition and ensure data integrity, we will maintain a copy of your Fram data as of 21st November 2025, ie: from `/cluster/projects` and `$HOME` directories on NIRD.

### What is copied on NIRD?

* **Projects:** Data under `/cluster/projects/` for projects with an active compute quota since allocation period 2024.2. 
* **Home Directories:** All users' `$HOME` folders.
* **Shared Areas:** Specific `/cluster/shared` areas. (Affected users have been contacted directly).

### How to access your copy of Fram data

The copy of Fram data is available as **read-only** in the following locations:

* **On Olivia this will be available from 22nd November 2025:**
    * Projects: `/nird/backup/hpc/fram/projects`
    * $HOME: `/nird/backup/fram/home`

```{note}
IMPORTANT: Copy of your Fram data on NIRD is temporary. The data in the locations mentioned above will be retained for one year (30th November 2026). 
If you have data on Fram that you wish to keep permanently, it is your responsibility to copy any data you wish to retain from this temporary location to an alternative, permanent storage solution.
Please note that NIRD is an available option for long-term data storage, and you are encouraged to migrate your Fram data to your NIRD project (either on Data Lake or Data Peak). 
All Olivia HPC projects are required to have an resource allocation on NIRD. You have an option to apply for NIRD project (NIRD Data Peak and Data Lake) if you don´t have already one.
If you have questions, please contact us via contact@sigma2.no.
```

---

## Your New Compute Resource: Olivia

### Access and Quota

Your compute quota from Fram has been administratively duplicated to Olivia. If you had access to Fram, you have access to Olivia now.

You can log in immediately:
```bash
ssh <username>@olivia.sigma2.no