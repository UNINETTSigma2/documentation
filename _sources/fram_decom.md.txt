---
orphan: true
---


(fram_decom)=

# Fram decommissioning information

**Published:** 2025-11-13

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

To facilitate the transition and ensure data integrity, we will maintain a copy of your Fram data from `/cluster/projects` and `$HOME` directories on NIRD.

### What is copied on NIRD?

* **Projects:** Data under `/cluster/projects/` for projects with an active compute quota since allocation period 2024.2. 
* **Home Directories:** All users' `$HOME` folders.
* **Shared Areas:** Specific `/cluster/shared` areas. (Affected users have been contacted directly).

### How to access your copy of Fram data

The copy of Fram data is available as **read-only** in the following locations:

* **On Fram (please use this to check integrity):**
    * Projects: `/cluster/backup/hpc/`
    * $HOME: `/cluster/backup/home`

* **On Olivia (available now on node `svc01`):**
    * From Olivia login nodes: first `ssh svc01` then you will find your Fram data on the paths below.
    * Projects: `/nird/backup/hpc/fram/projects`
    * $HOME: `/nird/backup/fram/home`

```{note}
IMPORTANT: Copy of your Fram data on NIRD is temporary. The data in the locations mentioned above will be retained until **31 March 2026**. 
If you have data on Fram that you wish to keep permanently, it is your responsibility to copy any data you wish to retain from this temporary location to an alternative, permanent storage solution.
Please note that NIRD is an available option for long-term data storage, and you are encouraged to migrate your Fram data to your NIRD project (either on Data Lake or Data Peak). 
If you have questions, please contact us via contact@sigma2.no.
```

---

## Your New Compute Resource: Olivia

### Access and Compute quota

If your project _only_ had a compute quota on Fram, this has been administratively duplicated to Olivia. If you had access to Fram, you have access to Olivia now.

You can log in immediately:
```bash
ssh <username>@olivia.sigma2.no
```

Please contact contact@sigma2.no in the case you have trouble with this.

Note that HPC projects with a quota on more than one system will not automatically get a compute quota on Olivia. To apply for additional compute quota, the project PI must login to metacenter.no and send an extra application ("Apply to extend granted quota" from the applications menu).

## Disk quotas on Olivia
Projects migrating from Fram to Olivia will receive the same `/cluster/project` disk quota on Olivia as they currently have on Fram.

**Please note that this is a temporary exception**

To facilitate a smooth transition for active projects migrating from Fram to Olivia, a temporary exception has been made for your Olivia:/cluster/projects quota. Projects migrating from Fram will receive the **same disk quota on Olivia as they currently have on Fram.**
This exception is a temporary transitional arrangement and is valid until **31 March 2026**. These projects must plan to move their data to NIRD before this date.

# Changelog:
- 13 Nov: Added clarification in "Access and Compute quota"
- 10 Nov: Updated status of Olivia data path
- 6 Nov: Updated status of Olivia data path
- 4 Nov: Updated path of data on Fram (integrity check)
- 3 Nov: Added info regarding project disk quotas
- 3 Nov: Added this changelog