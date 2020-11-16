# Get ready to deploy a service through the NIRD Toolkit

Only the project leader or the executive officer of a NSxxxxK project can deploy applications through the NIRD Toolkit. 
The project leader/executive officer can also decide who else shall be authorized to deploy application through 
the NIRD Toolkit and who shall be able run the deployed application. 
This is done by creating a group in Dataporten and connecting it to the resources in MAS (NSxxxxK). 
Those members of the group who hold administrative rights will deploy applications, ordinary members will run applications.

Follow the step-by-step procedure below to create and administer your group. ***If you are from University of Oslo, {ref}`read here <access-uio>` .***

1. Go to [Dataporten](https://minside.dataporten.no) and select the institution you belong to from the drop-down menu. If your institution does not appear there (Feide login), then select "Feide guest" from the drop-down menu in the "Other alternative login". You will then be redirected to the OpenIDP page. Create an account in OpenIDP by following the procedure and, once the account has been created, use it to log in to Dataporten as Feide guest.

   ![Dataporten login](imgs/Login.png "Dataporten login")

2. Once logged in, you will be redirected to the Dataporten dashboard. Create a new group by clicking on the link on top of the page ("Create New Group"). 

   ![Dataporten dashboard](imgs/DataPorten-daskboard.png "Dataporten dashboard")


3. Once the group is created, visualize the "Details" of the newly created group. You will be redirected to a page visualizing information about the group, including the "Group ID" (fc:adhoc:xxxx-xxx-xxxxx-xxxxx).

   ![Dataporten Group-ID](imgs/Group-ID.png "Dataporten Group-ID")

   **Send the Group name, Group ID and preferred short name to sigma2@uninett.no to be authorized to deploy a service through the NIRD Toolkit. Please specify which of your NSxxxxK projects you want this group to have access to.**

4. You can now authorize other co-workers to run the deployed application, by adding them to the newly created group. Click on "Edit" and you will be redirected to a page that contains a "Share Link" session. Copy the link and send it by mail to the person that you want to invite into the group. 

   ![Dataporten share-link](imgs/share-link.png "Dataporten share-link")

   Once the invited person accepts the invitation, he/she will appear as a member in the group.

   ![Dataporten members](imgs/members.png "Dataporten members")


   (OBS.: in the example above the owner the group has now become the member of the group.) You can now click on the little icon on the right-hand side to make the member the administrator. Once the new member has been made administrator, he/she will be able to deploy application, not only run.

5. If you have logged in to Dataporten through Feide or Feide OpenIDP, then your FeideID shall be connected to your MAS account. Check your account details in [MAS here](https://www.metacenter.no/mas/user/profile) and verify that you are registered through your Feide e-mail.

6. In order to deploy the application you shall log in to the [NIRD Toolkit](https://apps.sigma2.no) via your Feide or your OpenIDP account, previously added to the group in Dataporten. Select the group previously created in Dataporten in the "Authorized Groups". Now you are able to run your application, which will be connected to the project area NSxxxxK.
 
(access-uio)=

##  Access to the NIRD Toolkit for the users affiliated to the University of Oslo
New regulations with regard to access to services through Feide identity has been applied by the University of Oslo (UiO) for Feide users affiliated with UiO. If you want to use the NIRD Toolkit and you are from UiO, get in contact with us at <sigma2@uninett.no> . 


