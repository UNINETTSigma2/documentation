(Science-Europe-template-Practical-Guide)=
# Science Europe template - Practical Guide


## How to use the guide

The Science Europe guidelines consist of six core requirements - Data Collection, Data Quality, Data Storage, Ethical and Legal requirements, Data Sharing and Long Term Preservation, Data Management - translated into 13 questions. The present guide provides for each question (highlighted in italics) a high level description of the required information (in bullet and point), followed by the formulation of the question as it appears in easyDMP (in bold). Finally, one or more examples are provided to offer practical guidance for the formulation of the answer.

The structure is therefore the following:
### Core requirement

#### *Question*
- Description:
-  ...
-  ... 
   
- **Questions:**

  - **This is the question as it appears in easyDMP**

    - **Example Responses:**

      - ..... 
      -  ...

**All the question marked with a (\*) are mandatory.**
## The core requirements

- {ref}`Administrative information <Administrative-information>`
- {ref}`Data description and collection or reuse of existing data <Data-description-and-collection-or-reuse-of-existing-data>`
- {ref}`Documentation and data quality <Documentation-and-data-quality>`
- {ref}`Storage and backup during the research process <Storage-and-backup-during-the-research-process>`
- {ref}`Legal and ethical requirements, codes of conduct <Legal-and-ethical-requirements-codes-of-conduct>`
- {ref}`Data sharing and long term preservation <Data-sharing-and-long-term-preservation>`
- {ref}`Data management responsibilities and resources <Data-management-responsibilities-and-resources>`

(Administrative-information)=
## Administrative information

- Provide information such as name of applicant, project number, funding programme, (version of DMP).
- **Questions:**
  - **Please give the title of the project.** *
    - **Example Response:**
      - The Example Project 
  - **Please give the project number, if known.**
    - **Example Response:**
      - Project 12345678
  - **Please give the funding programme.** *
    - Example Response:
      - The Norwegian Research Council

(Data-description-and-collection-or-reuse-of-existing-data)=
## Data description and collection or reuse of existing data

####  *How will new data be collected or produced and/or how will existing data be re-used?*

   - Explain which methodologies or software will be used if new data are collected or produced.
   - State any constraints on re-use of existing data if there are any.
   - Explain how data provenance will be documented.
   - Briefly state the reasons if the re-use of any existing data sources has been considered but discarded.
   - **Questions:**
     - **Describe how you plan to collect new or use existing data.** *
        - **Example Response:**
          - The project will use data from three sources:
            - Experimental data produced by an X-ray crystallography on our sample materials. The data will be collected at the European Synchrotron Radiation Facility (ESRF <https://www.esrf.eu/>) and will be processed at the facility using the CCP4 (<http://www.ccp4.ac.uk/>) suite of software. Information on when the data was produced, and the conditions will be recorded in electronic notebooks at the ESRF. ASCII Log files will be produced by the CCP4 processing (the log files produced standardised output that is described in CCP4 documentation).
            - Simulation data on the molecular dynamics of the sample material. Simulation data does exist, but we need to produce more simulation samples to increase the statistical accuracy. For the existing simulations, information on how the data were produced is recorded in log files that are maintained on central servers at institution X (<https://example.org/institution/simulationData>) that are publicly accessible. Information on the additional simulation produced by this project will be stored in ASCII log files. The information in simulation log files follows the structure described in the documentation at institution X. The reused simulation data and the simulation data produced by this project will be publicly accessible at no cost. All simulations are produced with the GROMACS program (<http://www.gromacs.org/>).
            - Images of the objects from which the samples were taken. The images will be taken with conventional digital photography. The images contain information on the location and date when they were taken. In addition, information on the conditions in which the photographs were taken will also be electronically recorded in an e-logbook.





#### *What data (for example the kind, formats, and volumes), will be collected or produced?*

   - Give details on the kind of data: for example numeric (databases, spreadsheets), textual (documents), image, audio, video, and/or mixed media.
   - Give details on the data format: the way in which the data is encoded for storage, often reflected by the filename extension (for example pdf, xls, doc, txt, or rdf).
   - Justify the use of certain formats. For example, decisions may be based on staff expertise within the host organisation, a preference for open formats, standards accepted by data repositories, widespread usage within the research community, or on the software or equipment that will be used.
   - Give preference to open and standard formats as they facilitate sharing and long-term re-use of data (several repositories provide lists of such ‘preferred formats’).
   - Give details on the volumes (they can be expressed in storage space required (bytes), and/or in numbers of objects, files, rows, and columns).
   - **Questions:**
     - **Describe how much data, and the type of data you plan to collect or produce.** *
        - **Example Response:**
            - The project will create (approximately):
                - 100 GB of high-resolution images in the standard JPEG (<https://en.wikipedia.org/wiki/JPEG>) format. The JPEG format is chosen as a wide variety of software programs can read this format.
                - 5 TB simulation data produced by the project will be stored in the GROMACS (GRO <http://manual.gromacs.org/archive/5.0.3/online/gro.html>) format that is standard in the community and is open. In addition, we will use 2 TB of existing simulation data. So, a total of 7 TB simulation data.
                - 10 TB of experimental data produced by the CCP4 program and stored in the standardised Crystallographic Information File (CIF <https://en.wikipedia.org/wiki/Crystallographic_Information_File>) format that has an open license.

(Documentation-and-data-quality)=
## Documentation and data quality

####  *What metadata and documentation (for example the methodology of data collection and way of organising data) will accompany the data?*

   - Indicate which metadata will be provided to help others identify and discover the data.
   - Indicate which metadata standards (for example DDI, TEI, EML, MARC, CMDI) will be used.
   _ Use community metadata standards where these are in place.
   - Indicate how the data will be organised during the project, mentioning for example conventions, version control, and folder structures. Consistent, well-ordered research data will be easier to find, understand, and re-use.
   - Consider what other documentation is needed to enable re-use. This may include information on the methodology used to collect the data, analytical and procedural information, definitions of variables, units of measurement, and so on.
   - Consider how this information will be captured and where it will be recorded for example in a database with links to each item, a ‘readme’ text file, file headers, code books, or lab notebooks.
   - **Questions:**
        - **Describe how you will organise and document your data.** *
            - **Example Response:**
                - The experimental data will be arranged in directories according to sample, synchrotron beam run and processing run. Information will be recorded in a relational database at the project leader’s institution with access control that enables collaborators to access the information. The metadata schema will be extracted from the ESRF ICAT metadata catalogue and will follow the ICAT schema that is an agreed standard and contains all the information necessary to understand the data.
                - The simulation data will be stored in the central repository in institution X and will follow their layout. Metadata information will follow the metadata schema adopted by institution X which is used by many projects in this field. Documentation on the schema is widely available to researchers in the field and enables use of the data.
                - The e-logbook information on the digital photographs will follow the Dublin Core metadata standard (<http://www.dublincore.org/specifications/dublin-core/dcmi-terms/#section-3>) to record information on the images. Each image will have a unique identifier that will match the Dublin Core identifier term making it easy for researchers to match the metadata to the data.
                - Tutorials and documentation on the tools necessary to analyse the data will be maintained on the project web-site. In some cases, these will be links to widely-used tools.



####  *What data quality control measures will be used?*

   - Explain how the consistency and quality of data collection will be controlled and documented. This may include processes such as calibration, repeated samples or measurements, standardised data capture, data entry validation, peer review of data, or representation with controlled vocabularies.
   - **Questions:**
     - **Describe how you will control the consistency and quality of your data.** *
        - **Example Response:**
            - For the experimental data the quality of the data is recorded in the ESRF ICAT metadata catalogue and will be replicated to the project metadata catalogue. This metadata contains information on the position of the sample, the experimental station, beam conditions etc which is sufficient to understand the experimental data. The sample itself will be labelled and kept at the project leader’s institution for reference.
            - Simulation data quality will be recorded in log files and reference data will be produced during each simulation that will be compared with existing reference data to ensure simulations remain within tolerance. Information on the machines the simulations ran on and when is recorded in the log files which will be archived at institution X.
            - Digital photographs will be visually inspected on-site by project collaborators that have the right collect the images to ensure the images are of sufficient quality. A checklist of features will be drawn-up by experts in digital photography and each image will require approval by the WP leader responsible for acquiring the images.

(Storage-and-backup-during-the-research-process)=
## Storage and backup during the research process

####  *How will data and metadata be stored and backed up during the research?*

   - Describe where the data will be stored and backed up during research activities and how often the backup will be performed. It is recommended to store data in least at two separate locations.
   - Give preference to the use of robust, managed storage with automatic backup, such as provided by IT support services of the home institution. Storing data on laptops, stand-alone hard drives, or external storage devices such as USB sticks is not recommended.
   - Explain how the data will be recovered in the event of an incident.
   - Explain who will have access to the data during the research and how access to data is controlled, especially in collaborative partnerships.
   - **Questions:**
     -  **Describe how you will securely store and back up and recover your data during your project.** *
        - **Example Response:**
            - The experimental data will initially be stored on the ESRF storage facility during data collection. The data will be subject to the ESRF storage and backup procedures (<https://www.esrf.eu/UsersAndScience/Experiments/MX/How_to_use_our_beamlines/Prepare_Your_Experiment/Backup>) that project members will run in accordance with advice from ESRF.  The ESRF provides access control which the project will control. Once the experiment has completed data will be transferred to the Norwegian Infrastructure for Research Data (NIRD) project storage where data backup is provided by NIRD. NIRD provides storage with access control, only project collaborators will be provided access to the NIRD storage for the project. In the case of NIRD and ESRF project storage data being lost the backup procedures provide restore capabilities where lost data can be identified and recovered.
            - The simulation data will be produced on the Norwegian High-Performance Computing Facility (FRAM) which will store data on the NIRD project storage to which NIRD applies backup and recovery procedures. During the production data will be subject to access control restrictions to project collaborators. Once the simulation data are validated it will be transferred to the central storage at institution X which provides safe long-term storage for simulation data. Backup and recovery of NIRD storage is provided by NIRD and institution X will enforce the same backup and recovery procedures for the project’s data.
            - The images will be transferred from digital camera to the University of Oslo Sensitive Data facility (TSD) as the data contain information of a personal sensitive nature in accordance to Norwegian privacy regulation. Data will be backed-up according to TSD policies and only approved researchers will have access to the images. The project has been approved by the Regional Ethical committee (REK SØ 2019/1234).



####  *How will data security and protection of sensitive data be taken care of during the research?*

   - Consider data protection, particularly if your data is sensitive for example containing personal data, politically sensitive information, or trade secrets. Describe the main risks and how these will be managed.
   - Explain which institutional data protection policies are in place.
   - **Questions:**
        - **If your project uses sensitive data describe how you will take care of data protection and security.**
            - **Example Response:**
                - The images collected are data objects that are of a sensitive nature and therefore are subject to Norwegian legislation for handling and managing sensitive data as implemented by the University of Oslo (<https://www.uio.no/english/for-employees/support/privacy-dataprotection/>). The image data objects will be encrypted and imported inside the TSD by using an encrypted protocol. The camera removable hard-drives containing the images will be scrubbed and destroyed once the data has been transferred to the TSD. Only authorized collaborators will be provided access to the images, and export from the TSD server will be by no means possible. The images will be anonymised in accordance with the sensitive data legislation before exporting them out from the secure TSD area.

(Legal-and-ethical-requirements-codes-of-conduct)=
## Legal and ethical requirements, codes of conduct

####  *If personal data are processed, how will compliance with legislation on personal data and on security be ensured?*

   - Ensure that when dealing with personal data, data protection laws (for example GDPR) are complied with:
   - Gain informed consent for preservation and/or sharing of personal data.
   - Consider anonymisation of personal data for preservation and/or sharing (truly anonymous data are no longer considered personal data).
   - Consider pseudonymisation of personal data (the main difference with anonymisation is that pseudonymisation is reversible).
   - Consider encryption which is seen as a special case of pseudonymisation (the encryption key must be stored separately from the data, for instance by a trusted third party).
   - Explain whether there is a managed access procedure in place for authorised users of personal data.
   - **Questions:**
        - **If your project uses personal data describe how you will ensure compliance with legislation on personal data and security.**
            - **Example Response:**
                - The project has been approved by the Regional Ethical Committee (REK SØ 2019/1234). Sensitive Personal data will be stored inside TSD and access to the data will be strictly controlled by the Project principal investigator.
                - The project will handle sensitive personal data containing information about individuals eating habitudes and health condition. Data collected through questionnaire will be pseudorandomized, and the key-file will be stored in a separate area inside TSD accessible only by the project principal investigator.
                - The personal data (including mail address and private addresses) handled in the project will be collected after the explicit consent of the data owner, according to the GDPR regulation. Data will be deleted or anonymised after a maximum of four weeks.





####  *How will other legal issues, such as intellectual property rights and ownership, be managed? What legislation is applicable?*

   - Explain who will be the owner of the data, meaning who will have the rights to control access:
   - Explain what access conditions will apply to the data? Will the data be openly accessible, or will there be access restrictions? In the latter case, which? Consider the use of data access and re-use licenses.
   - Make sure to cover these matters of rights to control access to data for multi-partner projects and multiple data owners, in the consortium agreement.
   - Indicate whether intellectual property rights (for example Database Directive, sui generis rights) are affected. If so, explain which and how will they be dealt with.
   - Indicate whether there are any restrictions on the re-use of third-party data.
   - **Questions:**
        - **Describe how you plan to address other legal issues such as intellectual property rights and ownership.** *
            - **Example Response:**
                - The project will abide by the University’s intellectual property rights policy (<https://www.uio.no/english/about/regulations/research/intellectual-property/uio-policy-for-intellectual-property-rights.pdf>), and the data are subject to no other IPR claims.
                - During the course of the project data will remain restricted access until publication of research results. The data used in the publication will be submitted to an archive such as the NIRD research data archive where it will be publicly accessible. A Creative Commons license BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) will apply to all the data.



####  *What ethical issues and codes of conduct are there, and how will they be taken into account?*

   - Consider whether ethical issues can affect how data are stored and transferred, who can see or use them, and how long they are kept. Demonstrate awareness of these aspects and respective planning.
   - Follow the national and international codes of conducts and institutional ethical guidelines, and check if ethical review (for example by an ethics committee) is required for data collection in the research project.
   - **Questions:**
        - **If your data are impacted by ethical issues and codes of conduct describe how you will take account of them.**
            - **Example Response:**
                - The project will abide by the recommendations described in the EU Ethics and data protection guidelines <http://ec.europa.eu/research/participants/data/ref/h2020/grants_manual/hi/ethics/h2020_hi_ethics-data-protection_en.pdf> to ensure the sensitive image data are correctly handled and will seek ethical review by the University of our plan for handling the sensitive image data.



(Data-sharing-and-long-term-preservation)=
## Data sharing and long term preservation

#### *How and when will data be shared? Are there possible restrictions to data sharing or embargo reasons?*

   - Explain how the data will be discoverable and shared (for example by depositing in a trustworthy data repository, indexed in a catalogue, use of a secure data service, direct handling of data requests, or use of another mechanism).
   - Outline the plan for data preservation and give information on how long the data will be retained.
   - Explain when the data will be made available. Indicate the expected timely release. Explain whether exclusive use of the data will be claimed and if so, why and for how long. Indicate whether data sharing will be postponed or restricted for example to publish, protect intellectual property, or seek patents.
   - Indicate who will be able to use the data. If it is necessary to restrict access to certain communities or to apply a data sharing agreement, explain how and why. Explain what action will be taken to overcome or to minimise restrictions.
   - **Questions:**
        - **Describe how and when you will share your data, and relevant information, including data you intend to preserve.** *
            - **Example Response:**
                - The project intends to use the NIRD Research Data Archive to store data, it and the community considers to be of lasting value. This will include data used in publications and data used to derive the results. The project will supply metadata to the archive that will be made publicly accessible and searchable. The archived data will be issued a DOI and made publicly accessible. The simulation data will be deposited in the institution X long-term repository along with the log files and metadata. It will be given an DOI by institution X and will be publicly accessible. The archived data will include documentation on the tools that can be used and how to use the data.
                - Data will be published in the NIRD archive and made publicly available after the relevant articles have been published. One year after the end of the project the remaining data will be published in the archive.
                - The images are sensitive and will only be accessible upon request from the TSD.
                - The project will nominate a data manager responsible for fielding questions on the published data.





####  *How will data for preservation be selected, and where data will be preserved long-term (for example a data repository or archive)?*

   - Indicate what data must be retained or destroyed for contractual, legal, or regulatory purposes.
   - Indicate how it will be decided what data to keep. Describe the data to be preserved long-term.
   - Explain the foreseeable research uses (and/ or users) for the data.
   - Indicate where the data will be deposited. If no established repository is proposed, demonstrate in the data management plan that the data can be curated effectively beyond the lifetime of the grant. It is recommended to demonstrate that the repositories policies and procedures (including any metadata standards, and costs involved) have been checked.





####  *What methods or software tools are needed to access and use data?*

   - Indicate whether potential users need specific tools to access and (re-)use the data. Consider the sustainability of software needed for accessing the data.
   - Indicate whether data will be shared via a repository, requests handled directly, or whether another mechanism will be used?



####  *How will the application of a unique and persistent identifier (such as a Digital Object Identifier (DOI)) to each data set be ensured?*

   - Explain how the data might be re-used in other contexts. Persistent identifiers should be applied so that data can be reliably and efficiently located and referred to. Persistent identifiers also help to track citations and re-use.
   - Indicate whether a persistent identifier for the data will be pursued. Typically, a trustworthy, long-term repository will provide a persistent identifier.
   - **Questions:**
        - **Describe how you will assign persistent identifiers to your data.** *
            - **Example Response:**
                - Publication of data in the NIRD archive will result in a DOI being issued for the data. Users interested in using the data will be able to discover the data through the publicly available metadata and download the data with the link provided. The dataset includes documentation on how to use the data. A contact person, the data manager, will be available to respond to queries about the data.

(Data-management-responsibilities-and-resources)=
## Data management responsibilities and resources

####  *Who (for example role, position, and institution) will be responsible for data management (i.e. the data steward)?*

   - Outline the roles and responsibilities for data management/stewardship activities for example data capture, metadata production, data quality, storage and backup, data archiving, and data sharing. Name responsible individual(s) where possible.
   - For collaborative projects, explain the co-ordination of data management responsibilities across partners.
   - Indicate who is responsible for implementing the DMP, and for ensuring it is reviewed and, if necessary, revised.
   - Consider regular updates of the DMP.
   - **Questions:**
        - **Describe who will be responsible for the management of your data.** *
            - **Example Response:**
                - The project identifies a work package tasked with project data management. The work package will be responsible for ensuring the data are prepared, securely stored, are of sufficient quality, metadata is collected and data are published once articles have been published. The data manager will be the work package leader and will be a member of the project steering board.
                - The data manager will be responsible for quarterly updates of the data management plan.



####  *What resources (for example financial and time) will be dedicated to data management and ensuring that data will be FAIR (Findable, Accessible, Interoperable, Re-usable)?*

   - Explain how the necessary resources (for example time) to prepare the data for sharing/preservation (data curation) have been costed in. Carefully consider and justify any resources needed to deliver the data. These may include storage costs, hardware, staff time, costs of preparing data for deposit, and repository charges.
   - Indicate whether additional resources will be needed to prepare data for deposit or to meet any charges from data repositories. If yes, explain how much is needed and how such costs will be covered.
   - **Questions:**
        - **Describe the resources that will be dedicated to the management of your data such that it follows the FAIR (Findable, Accessible, Interoperable, Reusable) principles.** *
            - **Example Response:**
                - The project has factored into the project timeline the data management through the inclusion of the data management work package that includes personnel funded by the project. The data manager will be a permanent member of staff who will be able to field questions on the data once the project has concluded. The project proposal also includes a request for funds for NIRD and TSD storage to be used during the lifetime of the project. The NIRD archive where the published data will be stored, and the Institution X repository for the simulation data are free to use.

