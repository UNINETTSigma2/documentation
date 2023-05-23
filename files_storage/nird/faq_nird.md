# Frequently asked questions on NIRD

The FAQ on NIRD is prepared based on the Q&A session during the  {ref}`training-2023-spring-second-generation-nird` training event.

1. I am an HPC user. Should I apply for HPC storage or NIRD storage?
 
   	- higly depends on user need. But if you would like to use services like Toolkit, you need a nird storage project. Pls see the service description here.

   	- HPC storage is expensive, prefer NIRD for large volumes (15-20 TB), security features and  backups

2. Can you connect multiple projects to a single application? For instance, we have CMIP  data stored in different projects at the moment, and would like to compare across the different projects.

   	- yes it is possible if the PI’s (Project leaders) agreed to have common application.


3. What is the network bandwidth between new NIRD and Sigma2 HPC systems?

  	- the link which is provided between HPC system (Betzy,Fram,Saga) and new nird is 100Gbit/s
 
  	- 200GbE is between nird and service platform.

4. What is the difference between tier storage and datalake?
  
  	- Please see the detailed description [here](https://documentation.sigma2.no/files_storage/nird/ts_dl.html)
  	
	- You can also watch the training videos [here](https://www.youtube.com/watch?v=iBwhDsZtAzs&t=74s) from 48:18 minutes..

5. We have several instruments deployed in remote area. The instruments synchronize collected data with NIRD storage. Would it be possible do obtain dedicated ‘user accounts’ for such devices (to avoid using real user’s accounts e.g. for generating ssh keys)?

   	- The way forward now is to give S3 access and authenticate the user for. The S3 functionality is not still in production.

	- with minio it is possible for the owner of the service to create dedicated service accounts with specific access to some folders in the project connected to the minio service.

6. What does multitenancy mean?

   	- It means that each user does not get dedicated hardware to run on. The software we run serve several users on the same system. The tenants are logically isolated, but using the same physical system. See more [here](https://en.wikipedia.org/wiki/Multitenancy).

7. How is the data distributed between flash and disk based storage? Automatically (e.g. by access patterns) or manually?

	- We have set of policies which will take care of distribution automatically,and it is also possible to manually pin the file/folder on different tier. Pinning of data to specific tier is based on operational requirements. (NB: even the slowest tier is more capable than the Betzy high-performance storage.)

8. What are your recommendations to migrate our home directories from the old to the new NIRD? Should we copy the whole home directory with all hidden files or just our folders? Moreover, I have Miniconda installed in my home directory. Is it better to install it again on the new NIRD or is a copy of the entire old home directory fine?

   	- Please see the information [here](https://documentation.sigma2.no/files_storage/nird/old_nird_access.html)

   	- You don’t need to copy the whole home directory. You can choose what do you want to migrate. We suggest to not copy any conda files.

9. Are snapshots counted towards quota?

  	- No, snapshots are not counted on quota. Please remember that snapshots are temporary back up. Please see [here](https://documentation.sigma2.no/files_storage/nird/snapshots_lmd.html)

10. How does rclone compare to using rsync?

   	- We are not recommending rclone yet, but you can find documentation [here](https://www.clusterednetworks.com/blog/post/rsync-vs-rclone-what-are-differences)

11. where is the ‘conda’ installed? 

   	- you can load the module `ml Miniconda3/4.7.10`

12. We are currently using MinIO to provide an S3-compatible interface to our data on NIRD. Should we be considering the new NIRD object storage/Swift as an alternative? Any advantages or disadvantages?

   	- It is an option you can use if you have allocation on datalake. We haven’t done the benchmark test yet to say the advantages or disadvantages between minio and S3.

13. Is S3 actually going to replace the minio bucket service you were providing?

   	- No it is not going to replace minio. It is an additional service on NIRD.

15. As i understood all storage will be placed in a single location in the future, if yes, what will be the impact on high availability concept?

   	- The two storage clusters, TS and DL, are physically separated with redundant power, cooling lines, automated fire extinguishers.

16. Can I start a kernel at nird, which I can access from my local programming environment (e.g. spyder) to access and work with the data stored at my project area? Or how to acces the data from my local programming session?

   	- You can mount your NIRD project on your local machine using sshfs although it has some performance limitations.

   	- It is also possible to deploy a dedicated webdav service on the service platform. You can then mount it locally.

17. How to use underlying k8s and kubectl integration with NIRD resources?

   	- It is possible, you can contact us via support@nris.no

18. Could you please share the slides (.pdf) of the nird training?

   	- You can access the slides from [here](https://drive.google.com/drive/u/0/folders/1uevX2-bm9S7SePHQC6YUrWrO6J4lDfCA)

19. Is the NIRD training videos recording available?

   	- Yes, you can find it [here](https://www.youtube.com/watch?v=iBwhDsZtAzs&t=74s)

20. What are the  optimal storage structure in the shared project area and access by users.

   	- Data can be shared between the project with an agreement with the PIs.
   	- web service connected with projects can be used for external users
   	- minio and S3 are other options

21. Is there a possibility of running calculations on NIRD? 

   	- High performance computing like climate simulations are for HPCs
	
	- You can run post processing on NIRD, also see the [NIRD Toolkit documentation](https://documentation.sigma2.no/nird_toolkit/overview.html).

22. As I understood all storage will be placed in a single location in the future, if yes, what will be the impact on high availability concept?

   	- The two storage clusters, TS and DL, are physically separated with redundant power, cooling lines, automated fire extinguishers.

 
