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

## Data Migration and Backup

To facilitate the transition, Sigma2 has re-enabled backup for Fram `/cluster/projects` and `$HOME` directories.

### What is Backed Up?

* **Projects:** All project data for projects with an active compute quota since allocation period 2024.2.
* **Home Directories:** All users' `$HOME` folders.
* **Shared Areas:** Specific `/cluster/shared` areas. (Affected users have been contacted directly).

### How to Access Your Backup

Your backed-up data is available as **read-only** in the following locations:

* **On Fram (Available Now):**
    * Projects: `/cluster/backup/hpc/fram`
    * $HOME: `/cluster/backup/home`

* **On Olivia (Expected by 31 Oct 2025):**
    * Access to this backup will be enabled on the Olivia login nodes no later than Friday, 31 October 2025.

```{danger}
CRITICAL: Backup is Temporary
The backed-up data at the locations above will be kept only until **31 March 2026**.

If you have data on Fram you would like to keep permanently, it is **your responsibility** to copy any data you wish to keep from this backup location to an alternative, permanent storage solution before this date. Please set a calendar reminder.
```
---

## Your New Compute Resource: Olivia

### Access and Quota

Your compute quota from Fram has been administratively duplicated to Olivia. If you had access to Fram, you have access to Olivia now.

You can log in immediately:
```bash
ssh <username>@olivia.sigma2.no