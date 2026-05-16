# MODERN PATHOLOGY

![USCAP United States and Canadian Academy of Pathology logo](page_1_image_1_v2.jpg)

Journal homepage: https://modernpathology.org/

Review Article

# Future of Artificial Intelligence—Machine Learning Trends in Pathology and Medicine

Matthew G. Hanna<sup>a,b,*</sup>, Liron Pantanowitz<sup>a,b</sup>, Rajesh Dash<sup>c</sup>, James H. Harrison<sup>d</sup>, Mustafa Deebajah<sup>e</sup>, Joshua Pantanowitz<sup>f</sup>, Hooman H. Rashidi<sup>a,b,*</sup>

<sup>a</sup> Department of Pathology, University of Pittsburgh Medical Center, Pittsburgh, Pennsylvania; <sup>b</sup> Computational Pathology and AI Center of Excellence, University of Pittsburgh, Pittsburgh, Pennsylvania; <sup>c</sup> Department of Pathology, Duke University, Durham, North Carolina; <sup>d</sup> Department of Pathology, University of Virginia, Charlottesville, Virginia; <sup>e</sup> Department of Pathology, Cleveland Clinic, Cleveland, Ohio; <sup>f</sup> University of Pittsburgh Medical School, Pittsburgh, Pennsylvania

## ARTICLE INFO

## ABSTRACT

Article history:
Received 18 August 2024
Revised 19 December 2024
Accepted 1 January 2025
Available online 4 January 2025

Artificial intelligence (AI) and machine learning (ML) are transforming the field of medicine. Health care organizations are now starting to establish management strategies for integrating such platforms (AI-ML toolsets) that leverage the computational power of advanced algorithms to analyze data and to provide better insights that ultimately translate to enhanced clinical decision-making and improved patient outcomes. Emerging AI-ML platforms and trends in pathology and medicine are reshaping the field by offering innovative solutions to enhance diagnostic accuracy, operational workflows, clinical decision support, and clinical outcomes. These tools are also increasingly valuable in pathology research in which they contribute to automated image analysis, biomarker discovery, drug development, clinical trials, and predictive analytics. Other related trends include the adoption of ML operations for managing models in clinical settings, the application of multimodal and multiagent AI to utilize diverse data sources, expedited translational research, and virtualized education for training and simulation. As the final chapter of our AI educational series, this review article delves into the current adoption, future directions, and transformative potential of AI-ML platforms in pathology and medicine, discussing their applications, benefits, challenges, and future perspectives.

Keywords:
artificial intelligence
computational pathology
machine learning
operations

© 2025 THE AUTHORS. Published by Elsevier Inc. on behalf of the United States & Canadian Academy of Pathology. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).

## Introduction

In recent years, the proliferation of data and advancements in computational capabilities have catalyzed the rapid growth of machine learning (ML) technologies across health care.<sup>1-3</sup> The integration of ML into pathology and medicine has unlocked new possibilities for enhancing diagnostic accuracy, streamlining

laboratory operations, and improving patient care. Development, deployment, and management of these new ML systems, however, require a range of compatible tools and hardware that can be challenging and inefficient to assemble and use on a manual basis. ML platforms are starting to solve this problem by providing an integrated software, hardware, and process framework that supports development and deployment of ML models for multiple purposes, at scale. These platforms utilize computational pipelines and sophisticated algorithms that standardize and automate the stages of the ML model life cycle, from data handling and model development through validation, deployment, and performance monitoring. ML models developed on a standardized

\* Corresponding authors
E-mail addresses: hannamg3@upmc.edu (M.G. Hanna), rashidihh@upmc.edu (H.H. Rashidi).

---

# Your Journey Through This 7-Part Review Article Series

![Your journey through this 7-part review article series. 1) Introduction to AI-ML, 2) Generative AI in Medicine, 3) Non-Generative AI-ML in Medicine, 4) Statistics of ML in Medicine, 5) Regulatory Aspects of AI-ML in Medicine, 6) Ethical and Bias Considerations of AI-ML, 7) The Future of AI-ML in Medicine. You are here.](page_2_image_2_v2.jpg)

**Figure 1.**

Your journey through this 7-part review article series. AI, artificial intelligence; ML, machine learning.

platform can be deployed for multiple uses, for example, case management systems or digital pathology viewing software, with assurance of a well-documented build strategy and performance evaluation (Fig. 1).

From an operational perspective, an ML platform serves as a centralized ecosystem in which data scientists, engineers, and analysts, among other stakeholders (eg, developers, physicians, regulatory specialists, etc), collaborate to harness data-driven insights effectively. These platforms typically integrate a suite of tools and services designed to streamline the entire ML life cycle, encompassing data preparation, model training, evaluation, deployment, integration, monitoring, and feedback. Currently, artificial intelligence (AI) platforms are used in various areas of health care. Most notably, they are revolutionizing medical imaging analysis and interpretation. The AI-ML platforms enable deployment of ML applications such as (semi)automated analysis of medical images (eg, whole-slide images [WSI], dermoscopy, ophthalmology, X-rays, computed tomography scans, magnetic resonance imaging scans, etc) to assist with detecting abnormalities, diagnosing diseases, and also predicting normal or benign conditions.<sup>4-8</sup> AI-ML applications designed for pathology include sophisticated techniques for image segmentation and quantification in pathology to aid pathologists in identifying and quantifying tissue structures, cell types, and biomarkers with higher precision, increased standardization, and improving productivity.<sup>9-13</sup> Integrating AI-ML into clinical decision support systems enhances

diagnostic accuracy and treatment planning. These platforms analyze clinical data in real time, providing health care professionals with evidence-based recommendations and alerts for potential anomalies (Fig. 2).

Such AI-ML platforms can also help in advancing our personalized medicine initiatives, which are set to analyze patient data in tailoring treatment plans and therapies based on individual genetic profiles, biomarkers, and disease characteristics. These can ultimately lead to more effective and targeted health care interventions. Additionally, the rise of wearable devices and "Internet of Things" sensors has further enabled the continuous monitoring of outpatient health metrics (ie, daily routine activities outside of hospital clinics) through various ML platforms.<sup>14-19</sup> Such AI-ML platforms may help further integrate with the electronic health record (EHR) systems to enhance data interoperability, automate documentation tasks, and provide comprehensive patient insights to our health care providers at the point of care.<sup>20</sup>

The increased integrations of AI-ML platforms into our medical operations will also emphasize and eventually mandate collaborations between data scientists or data science literate clinicians and our frontline health care professionals (ie, providers), fostering a synergistic approach to the future of patient care.<sup>21</sup> This collaborative model will also help in enhancing diagnostic accuracy, streamlining workflows, and improving patient care outcomes within our future landscape.

# Machine Learning DevOps

![Machine Learning DevOps diagram showing three stages: Designing the ML Model, Developing the ML Model, and Putting the ML Model Into Operation.](page_2_image_1_v2.jpg)

* Ascertain criteria and requirements
* Prioritize ML applications
* Commercial offering
* Harvest data

* Prepare and process data
* Feature development
* Train model
* Experiment
* Analyze and reflect on performance

* Initial deployment
* Continuous Integration/Continuous Deployment (CI/CD) workflows
* Surveillance, monitoring (e.g., data drift assessment, etc.), and triggering

**Figure 2.**

ML model lifecycle: Research-model development, iteration, internal/external validation, clinical deployment, and ongoing monitoring. Many laboratories will select commercial offerings that will require laboratory verification or validation. However, in the development life cycle of a ML model, there is an iterative cycle for designing, developing, and operationalizing the model for its intended use. AI, artificial intelligence; ML, machine learning.

---

## Machine Learning Operations

Mature use of information technology (IT) in industry is recognized to encompass a life cycle that includes software/system development, deployment, operational management, and replacement. The integration of development and operations to manage this life cycle within an enterprise has become known as Dev-Ops. Analogous to that within the AI-ML space is the ML operations (ML-Ops), which encompasses a set of practices and tools used to deploy and manage the deployed ML models in the production setting (eg, routine clinical practice).<sup>22</sup> Similar to Dev-Ops, ML-Ops fosters collaboration and communication among data scientists, information technologists, subject matter experts, and administration (ie, operations leadership and project managers). ML-Ops plays a crucial role in health care by promoting integration of disparate expertise (physicians, data scientists, IT, security, and administration) for decision-making, and linkage of ML decision-making with patient outcomes. ML platforms are becoming foundational to ML-Ops by providing an effective collaborative environment for these stakeholders.<sup>23</sup>

ML-Ops acknowledges the importance of human expertise throughout the ML life cycle. Human-in-the-loop processes involve human oversight and intervention. For early model development, this may be seen with data labeling, model training, validation, interpreting model outputs, and ensuring ethical considerations in AI-driven decisions. However, for the deployment (clinical use) of such systems in a clinical laboratory setting, pathologists and other medical professionals need to remain the arbiters of such ML model outputs to ensure patient test results are accurate and precise. Future systems may allow for autonomous AI systems; however, these are relatively limited in the health care setting today.<sup>24</sup>

## Platform Components

An AI-ML platform is a comprehensive software framework that provides various tools, libraries, and capabilities to support the entire life cycle (eg, data collection, model development, training and optimization, deployment, and ongoing monitoring). A machine learning model is a mathematical representation or algorithm trained on data to identify patterns, make predictions, or solve specific tasks by generalizing from examples without being explicitly programmed for the task ("training" the algorithm) (Fig. 2). This includes the algorithm packaged with the learned patterns or relationships in the data that thereby enables predictions or decisions to be made. Effective ML begins with high-quality data that are representative of the real world. ML platforms need to facilitate data aggregation from diverse sources including production clinical systems (eg, laboratory information systems), while ensuring data consistency and quality. Data sets must be managed and correctly segmented into training data to create the model and test data using separate held-out data to evaluate their performance. The percent or size of training, test, and validation data sets should be distributed appropriately for the intended clinical model and to capture target data diversity, support subgroup analysis, and statistical power to demonstrate efficacy. ML platforms provide frameworks and libraries that enable data scientists to build and train models using diverse algorithms and techniques. These environments support experimentation with different model architectures and settings to optimize model performance, and validation of models using a

variety of performance metrics appropriate for the model context with sequestered test data. Once models are trained and validated, platforms supporting ML-Ops facilitate deployment of models into production environments.<sup>25</sup> This may include model version control, containerization, orchestration, continuous ML model performance measures, and interfacing with existing IT infrastructure to ensure scalability, reliability, and efficient use of computational resources. ML-Ops includes integration via standard application programming interfaces with existing IT infrastructure, potentially including local data storage systems (eg, archived WSIs) as well as cloud services. Monitoring of the performance of deployed models is critical to ensure continuing effective performance over time, and although less commonly done now, future ML platforms are likely to track model performance, detect data drift or shift, and alert for timely updates or retraining as necessary.<sup>26,27</sup>

## Standards

Broadly speaking, ML-Ops standardization in any enterprise involves adopting best practices, model development frameworks, and tools that are shared and well understood within the enterprise, facilitating collaboration, reproducibility, and scalability of ML workflows, and reducing errors. In addition to data handling and model development tools, the environment may include version control systems (eg, Git), containerization technologies (eg, Docker), and orchestration tools (eg, Kubernetes). ML platforms may also provide data and metadata mapping tools to facilitate deployment in local health care environments. Docker is a tool that packages applications into containers, ensuring consistent performance across different environments. Containers are lightweight, share the same operating system kernel, and use fewer resources compared with traditional virtual machines. Docker simplifies application deployment and scaling. Kubernetes manages and orchestrates Docker containers, automating their deployment, scaling, and management across clusters. Furthermore, an ML model that was trained on Digital Imaging and Communications in Medicine (DICOM)-formatted whole-slide hematoxylin and eosin (H&E)-stained images and developed to detect prostate cancer in a biopsy specimen will also need to be mapped as such within the laboratory information system as follows: tissue site=prostate, specimen=biopsy, stain=H&E, file type=DICOM. This ensures that the right ML model is assigned on the right specimen, the right stain, the right data set (eg, whole-slide scanner output), and the right patient. Currently, these mappings are manually developed at local hospital sites and tools supporting them must be site specific. Future developments in standard data elements and their contents (eg, standard vocabularies) will hopefully simplify mapping tools and decrease the cost, effort, and error associated with model (and other system) deployment.

For conventional laboratory medicine testing, manufacturer package inserts have been commonplace and may offer a framework for ML testing in the laboratory setting. Standardization of documentation for ML models with improved operational transparency is an important goal of ML-Ops. For example, model cards are standard structured documents that can be used to communicate key information about an ML model.<sup>28</sup> They are inspired by similar concepts such as package inserts that are well known in the laboratory medicine and pharmaceutic landscape. Such documentation is crucial to provide transparency, accountability, and context regarding the development, performance, and

---

Table
Representative components for ML model card in pathology


| Component                                | Description                                                                                                       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Model name                               | Unique identifier or name assigned to the ML model                                                                |
| Description                              | Brief summary describing the purpose and functionality of the ML model                                            |
| Pathology/laboratory domain              | Specific area of pathology or laboratory medicine the model addresses (eg, histopathology and clinical chemistry) |
| Disease/condition                        | Disease or medical condition targeted by the model (eg, cancer type, infectious disease)                          |
| Biological specimen                      | Type of biological specimen analyzed by the model (eg, tissue biopsy and blood sample)                            |
| Testing method                           | Laboratory testing method or technique used (eg, immunohistochemistry and PCR)                                    |
| Clinical end point                       | Clinical outcome or parameter predicted or assessed by the model (eg, diagnosis and prognosis)                    |
| Input data                               | Types and sources of data inputs used to train the model (eg, imaging data and genomic data)                      |
| Output                                   | Types of outputs generated by the model (eg, diagnostic report and risk score)                                    |
| Performance metrics                      | Evaluation metrics used to assess the model's performance (eg, sensitivity, specificity)                          |
| Training data set                        | Description of the data set used for training the model (eg, no. of samples and data sources)                     |
| Training duration                        | Time taken to train the model, measured in h or d                                                                 |
| Hardware requirements                    | Hardware specifications needed to deploy and run the model effectively (eg, GPU and RAM)                          |
| Software dependencies                    | Software libraries and frameworks required for model execution (eg, TensorFlow and PyTorch)                       |
| Model file location                      | Path or location where the model file is stored for deployment                                                    |
| Deployment environment                   | Target environment in which the model is deployed (eg, hospital network and cloud platform)                       |
| Interfaces/APIs                          | APIs or interfaces for integrating the model with clinical systems or workflow                                    |
| Regulatory compliance                    | Compliance with regulatory standards (eg, FDA and CLIA) applicable to clinical use                                |
| Security measures                        | Security protocols and measures to protect patient data and model integrity                                       |
| Maintenance schedule                     | Schedule for model updates, monitoring, and maintenance                                                           |
| Ethical considerations                   | Ethical considerations regarding model use and patient impact                                                     |
| Documentation                            | Detailed documentation including model architecture, training procedures, and usage guidelines                    |
| Verification/validation methods and data | Methods and data sets used for model validation to ensure reliability and accuracy                                |
| Notes/comments                           | Additional relevant information or comments about the model and its deployment                                    |



API, application programming interface; CLIA, Clinical Laboratory Improvement Amendments; FDA, Food and Drug Administration; GPU, graphics processing unit; ML, machine learning; RAM, random access memory.

potential limitations of ML models used in pathology and laboratory medicine.<sup>28-31</sup> Some ML platforms automate standard documentation of models developed within the platform in a model card–like format, including a data description, data processing and segmentation, type of model, training environment and parameters, validation method and metrics, and model performance, as well as allow additional human editing for project-specific details (Table).

## Deployment Strategy

Operationalizing an ML model presents a pivotal decision around certain deployment strategies (eg, on-premise, cloud, and edge-computing) and implementation frameworks. On-premise (ie, local/private) deployment generally offers the advantage of direct control over data security and compliance, particularly for sensitive patient information under regulatory frameworks such as the Health Insurance Portability and Accountability Act or the General Data Protection Regulation. This does not inherently make on-premise deployment more secure than other models; instead, security depends on proper implementation of best practices for the chosen deployment strategy. On-premise deployments allow for tailored customization and integration into the historical IT infrastructure to align with specific institutional workflows, potentially optimizing latency for real-time diagnostic applications within localized networks. However, they necessitate investments in hardware and ongoing maintenance costs, posing challenges for institutions with limited resources or scalability needs.

Cloud deployments on the other hand offer scalability and elasticity through rapid and flexible provisioning of computational resources and managed services from providers (eg, autoscaling, managed databases, and AI/ML tools). Cloud solutions

have varying cost models, with data tiering differences based on resource allocations, latency, and the frequency of access. Although cloud deployments introduce unique security considerations owing to their architecture (eg, added exposed services), they can achieve strong security levels through proper safeguards, such as encryption, access controls, and regular audits. For either on-premise or cloud deployments, cybersecurity and concerns regarding data privacy and regulatory compliance are warranted.<sup>32</sup> Given the sensitivity of data and models, ML platforms should also prioritize security features such as access controls, encryption, and audit trails.

Last, edge computing has started to emerge as an efficient process to compute in tandem as clinical data (eg, WSIs, white blood cell differentials, and chemistry test results) are being generated. Although edge computing can minimize latency and enable localized processing of data, its advantages depend heavily on specific use cases and deployment contexts. Being "closer" to the data may be beneficial for certain real-time applications; however, there may be additional challenges related to hardware accessibility and scalability. By analyzing high-resolution images and diverse data types locally, edge computing has the potential to improve diagnostic efficiency and support compliance with data privacy regulations by keeping data in-network. However, these benefits require additional infrastructure and careful planning to mitigate complexities and ensure seamless integration into existing IT frameworks. In short, edge computing is about decentralizing computing to be closer to the data source, whereas on-premise and cloud computing focus on centralized or elastic infrastructure, respectively. Implementation of best practices must be adhered to mitigate the risks and challenges of each deployment type. Overall, each ML deployment strategy offers distinct advantages and considerations based on one's institutional needs, resources, security, compliance, bylaws, and regulatory requirements.<sup>33,34</sup>

---

## Reporting, Integration, and Workflow

Typically, an ML-Ops framework will provide reports that include metrics related to ML model performance and generalizability, operational efficiency, and business impact. These insights may include but are not limited to key metrics such as model accuracy, inference latency, resource utilization, and certain return on investment (ROI) measures. For instance, such a rigorous process of performance evaluation is essential for many of our anatomical and clinical laboratory tests, as mandated by regulatory standards such as the Clinical Laboratory Improvement Amendments of 1988.<sup>35</sup> Pathology departments will have to adhere meticulously to ML developer/manufacturer guidelines, regulatory directives, and established best practices to ensure the accuracy and reliability of the AI-ML model in use.<sup>36</sup> Central to this process is the evaluation of AI-ML models to ascertain their ability to generalize predictions across diverse data sets, despite potential disparities in data representation, distribution, or interlaboratory variations. Reporting on these metrics can facilitate ML-Ops for each model intended to be deployed clinically. Once a model is verified and/or validated using local data and deemed ready for clinical use, integration into the various clinical systems will help drive adoption and alleviate pathologist frustration when using the model in clinical practice. ML-Ops also emphasizes interoperability between different components of the ML pipeline (data storage, model training, and deployment) and integration with existing clinical enterprise systems (eg, EHR, laboratory information system, and image/case management system). Interoperability ensures seamless data exchange, compatibility across systems, and positive human-computer interactions.

## Business Analytics

Measuring ROI involves assessing several critical metrics that gauge the effectiveness and strategic alignment of AI-ML initiatives within an organization. Organizations may measure ROI in ML-Ops by assessing improvements in patient care-related outcomes, operational efficiency, cost savings, and business outcomes derived from deploying ML models. Key factors influencing ROI include improved diagnostic accuracy when using validated ML solutions, improved generalizability, and a reduced time from data generation to model output availability to a pathologist and other medical professionals, enabling automation and scalability of ML applications. More specifically, time to deployment is a unique metric that evaluates the efficiency of transitioning from model development to deployment in production environments. A reduced time to deployment indicates enhanced agility and responsiveness in delivering AI solutions to address medical diagnostic or business needs. Model performance metrics, including accuracy, precision (ie, positive predictive value), recall (ie, sensitivity), and other performance indicators, signify the quality of decision-making capabilities for ML models. However, classical performance metrics alone are often insufficient to fully characterize model performance, particularly when assessing resilience to data drift, bias, and generalization across diverse data sets. These data sets may have disparities in data representation, distribution, or interlaboratory variations, posing challenges to model reliability. A comprehensive model evaluation paradigm should include analyses of underrepresented data subgroups and discordant samples to ensure fairness and robustness. Improved accuracy, coupled with model transparency, robust monitoring, and fairness evaluations, may not only enhance patient care but also  contribute to practitioner satisfaction and trust in deploying ML models. Cost metrics focus on optimizing resource allocation and scaling strategies, thereby reducing operational expenses associated with computing resources and data processing infrastructure. Quantifiable business impact metrics such as revenue growth, cost reduction, and customer satisfaction serve as direct indicators of how ML platforms contribute to organizational success and market competitiveness. Additionally, future trainees may be more inclined to match at institutions with computational programs, further breeding higher caliber trainees who positively affect the academic program and act as a substrate for local faculty appointments. Strategic considerations in achieving positive ROI include aligning AI initiatives with overarching business goals, investing in talent development and retention, optimizing infrastructure costs, and continuously monitoring and optimizing ML workflows based on patient care and business metrics. By prioritizing these metrics, organizations can effectively measure and maximize the value generated by their investments in ML-Ops, fostering innovation and sustainable growth in a data-driven academic and business environment.

## Multimodal Artificial Intelligence

Multimodal AI refers to an AI-ML system that integrates diverse data types, such as medical imaging (eg, WSIs and magnetic resonance imaging scans), genomic data (eg, DNA sequencing), and clinical data (eg, patient demographics and laboratory results), to enhance decision-making in health care.<sup>37</sup> Integrating clinical data with multiple types of data further enriches this analysis by contextualizing findings within the patient’s medical history, medications, and comorbidities, thereby supporting holistic patient management strategies. Multimodal AI offers several advantages in pathology and laboratory medicine over unimodal models. Multimodal AI models often require less data, can better handle more context-intensive tasks, and offer enhanced diagnostic accuracy by leveraging complementary information from different data sources. For instance, abnormalities identified on imaging can be correlated with genomic markers associated with disease susceptibility or progression, which can translate to a more capable and useful outcome in the decision support pipeline. Multimodal AI thus holds tremendous potential in enhancing our current diagnostic practices. Notably, even when AI is integrated into our current workflow, it is typically a single-modal (ie, single data type) ML model approach that is useful but may also not provide a comprehensive picture to our practitioners. By integrating multiple data streams and dimensions of patient health, AI systems can identify subtle patterns and correlations that may not be apparent through unimodal analyses alone. For instance, combining histopathologic images with genomic data (ie, DNA or RNA) can provide further insights into the underlying disease progression by incorporating such molecular mechanisms that can ultimately lead to a more precise diagnosis and prognostic measure. Chen et al<sup>38</sup> describe a multimodal deep learning approach that integrated the analysis of pathology WSIs with molecular profile data across 14 cancer types. Their algorithm adeptly combines these diverse data modalities to predict patient outcomes and reveals prognostic features associated with both favorable and unfavorable clinical results. This comprehensive approach reduces diagnostic errors and variability among pathologists, thereby enhancing the reliability and reproducibility of clinical diagnoses. Multimodal AI also enhances diagnostic speed by automating the fusion and analysis of disparate data, reducing

---

the time required for comprehensive evaluation and decision-making.<sup>39,40</sup> This acceleration is crucial in time-sensitive clinical scenarios, such as intraoperative consultations, cancer diagnoses, or infectious disease identifications. In addition to improving diagnostic accuracy, multimodal AI enables the delivery of personalized treatment recommendations tailored to individual patient profiles. By integrating genomic data, imaging findings, and clinical parameters, AI algorithms can stratify patients into subgroups based on their disease risk, prognosis, and predicted response to therapy. This precision medicine approach empowers patient-facing health care providers to optimize treatment strategies, minimize adverse effects, and improve patient outcomes. AI-driven decision support systems can suggest targeted therapies or clinical trials based on a patient's demographics, disease histomorphology, genetic mutations, and disease stage, facilitating timely and data-driven treatment decisions. A more personalized approach to patient care improves patient outcomes through more precise disease modeling and better predictive response to therapies, while achieving higher patient satisfaction and optimizing resource allocation in health care settings.<sup>37</sup>

## Multiagent Frameworks

Multiagent frameworks represent a sophisticated approach in which multiple AI models interact directly with one another to achieve complex tasks and optimize decision-making processes.<sup>41-44</sup> These frameworks (eg, AutoGen and crewAI) are designed to mimic collaborative behaviors seen in human teams, in which each agent contributes specialized knowledge and skills to a shared objective. These frameworks are designed to facilitate the collaboration of multiple AI agents, each with their unique capabilities and knowledge bases, to solve complex problems that may be beyond the capacity of a single AI system. These frameworks differ fundamentally from traditional or single-model approaches by offering diverse perspectives on identical input data, enabling a deeper and more nuanced understanding of problems that a single model or representation may fail to capture. By incorporating agents with unique capabilities and knowledge bases, multiagent systems provide a framework for addressing challenges that require multidisciplinary insights and dynamic decision-making. It is important to distinguish multiagent frameworks from multimodal model deployments. While multimodal systems aim to integrate diverse data types (eg, imaging, text, and genomic data) into a single model to provide holistic predictions, multiagent frameworks involve multiple independent agents that may analyze the same data but from distinct perspectives or using distinct methodologies, collaborating to arrive at more robust solutions. While multimodal models aim to consolidate and process heterogeneous data sources into unified insights, multiagent systems excel at breaking down complex problems into specialized tasks handled by independent agents, fostering adaptability and modularity in decision-making processes. Specialized AI agents can perform specific tasks such as image analysis (eg, pathology and radiology), natural language processing (NLP) (eg, in clinical documentation or patient interaction), or predictive analytics (eg, for patient outcomes or resource allocation). Decision support AI agents can integrate multiple AI models to provide comprehensive decision support for clinicians, helping in diagnosis, treatment planning, and personalized medicine. Collaborative platforms facilitate communication and collaboration among health care professionals and AI systems, enabling interdisciplinary teamwork and optimized patient care. These frameworks can automate decision-making and workflow

optimization, and improve collaboration and communication among health care professionals.<sup>45</sup>

A promising application of multiagent frameworks in health care is in the area of clinical decision-making and workflow optimization. In a health care setting, there are numerous complex decisions that need to be made, often under time pressure and with incomplete information. In pathology, these decisions can range from diagnostic reports, reporting of immunohistochemical biomarkers, blood utilization, chemistry interferences, and determining the most effective treatment plan to managing the allocation of resources in a hospital. Multiagent frameworks can help to automate and optimize these decision-making processes. In pathology, multiagent frameworks could facilitate diagnostic workflows by combining different analytical capabilities, such as image analysis for whole-slide pathology, NLP for clinical documentation, and predictive analytics for patient outcomes. Unlike traditional systems that focus on individual tasks, these agents work collaboratively, often evaluating identical data from multiple angles. For example, a multimodal model may integrate pathology images and EHR data within a single system to provide a unified prediction, whereas a multiagent framework could involve one agent specialized in extracting quantitative image features, another in interpreting clinical text, and a third in correlating these findings to generate a more comprehensive diagnosis. This modularity allows for greater adaptability and specialization in multiagent systems compared with the static integration of multimodal approaches. For example, a multiagent system could be used to analyze a patient's medical history, symptoms, and test results, and then recommend a diagnosis and treatment plan. This in turn helps to reduce the workload of our health care professionals, improve the accuracy of our diagnoses, and further ensure that patients receive the most (cost)-effective treatment possible. Furthermore, multiagent frameworks can be used to optimize workflows in various areas of our medical arena. A multiagent system could also be used to manage the scheduling of provider's service days, optimize patient appointments based on the status of their current and past evaluation and reporting, and coordinate consultative care among different health care providers, to name a few. Another potential application of such multiagent frameworks in our health care landscape is improving collaboration and communication among pathologists or radiologists with patient-facing providers. In an optimized health care setting, effective collaboration and communication are critical not only in ensuring the quality of care that our patients receive but also in enhancing the well-being of our practitioners. However, our complex health care landscape with its diverse set of health care professionals along with the need for consistent, timely, and accurate information poses many hurdles and challenges that are currently hard to fully address. Multiagent frameworks can help with this by facilitating an environment that is conducive to an optimized sharing of information, coordination of actions, and a streamlined collective decision-making framework among our health care professionals. An example of this could also include a multiagent system that can aggregate multidisciplinary patient presentations (eg, tumor board) by incorporating imaging findings, diagnostic conditions, surgical outcomes, and treatment plans that help facilitate management decisions by all health care professionals involved in patient care. For more enhanced communications, a multiagent system could also be used to coordinate the actions of different health care professionals, ensuring that they work together effectively and efficiently to deliver multidisciplinary care. For example, the system could be used to automate report notifications to various health care

---

![Diagram illustrating Multi-agent (agentic) artificial intelligence integration across three layers: Inputs Layer (Medical History, Radiology Imaging, Patient Samples, Lifestyle Habits), Agents Layer (Pre-Analytical, Analytical, and Post-Analytical phases involving various personnel and automated machines), and Output Layer (Diagnosis, Treatment Plan, Final Report).](page_7_image_1_v2.jpg)

**Figure 3.**
Multi-agent (agentic) artificial intelligence integration.

stakeholders and generate personalized email communications discussing the prospective next steps for each medical domain, scheduling of appointments, allocation of resources, and coordination of care among different health care professionals. Multiagent frameworks could hence help improve the efficiency of health care delivery, reduce waiting times for patients, and improve the overall quality of care. Clearly, multiagent frameworks represent a transformative paradigm in health care by harnessing the collective intelligence of AI systems to automate decision-making, optimize workflows, and enhance collaboration among health care professionals. As these frameworks continue to evolve, their integration into clinical practice promises to redefine standards of care, accelerate medical innovation, and ultimately improve patient outcomes (Fig. 3).

# Artificial General Intelligence

Artificial general intelligence (AGI) refers to a type of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks that is at a level equivalent to that of a human being.<sup>46</sup> As of August 2024, AGI does not exist. However, the recent breakthroughs in the field of AI are demonstrating significant potential. Unlike narrow (or weak) AI, which comprises most of our current AI applications (ie, ML) and is designed to perform specific tasks (eg, such as a specific tumor detection task), AGI can theoretically perform any intellectual task that most human beings are able to do. Conceptually, AGI holds immense potential for transforming medicine.<sup>46</sup> Current cutting edge research and development efforts in the AI domains are focused on eventually creating AGI systems that can understand, learn, and apply knowledge across a wide range of tasks. The potential impact of AGI on medicine will be profound. It could revolutionize health

care delivery and change current standards of care. AGI could analyze a patient’s medical history, treatments, genetic data, wearable device medical data, lifestyle factors, pathology, and laboratory test results along with predicting disease risk and even suggesting preventative measures. It could also assist in complex surgical procedures, monitor the patient’s health in real time, and provide personalized health advice. Moreover, AGI could accelerate drug discovery by predicting the efficacy and safety of potential chemical candidates, thereby reducing the time and cost of bringing new therapeutic targets to market.

As noted, despite the promising potential of a future with AGI integrated, it is currently not a part of our AI arsenal of tools to employ. Most of the current AI applications in health care are examples of narrow AI, designed to perform specific tasks such as image analysis or predictive modeling. However as noted, there is a growing interest in developing AGI systems that can integrate and interpret diverse types of health data to support clinical decision-making. It is also important to keep in mind that developing AGI presents significant technical challenges, including the need for vast amounts of diverse and high-quality data, the difficulty of modeling complex human cognitive processes, and not to mention the major ethical considerations that would need to be addressed with AGI. The development of AGI will likely require breakthroughs in several areas of AI research, including ML, NLP, and cognitive modeling. Despite these challenges, the potential benefits of AGI in health care are unprecedented. By integrating and interpreting diverse types of health data, AGI could provide more accurate diagnoses, personalized treatment plans, and proactive health management. It could also help address some of the major challenges faced by health care today, such as the rising cost of care, the shortage of health care professionals, cross-discipline communication, and the need for more effective treatments for complex diseases. As mentioned, the development and use of AGI

---

in health care will also undoubtedly raise important ethical and societal issues. These include concerns about data privacy and security, the potential for job displacement, and the risk of exacerbating bias in health care. Therefore, before tackling the AGI space, it will be imperative to engage a wide range of stakeholders, including patients, patient advocacy organizations, health care providers, policymakers, and the public, in discussions about the development and use of such advanced frameworks within our health care domains.

# Virtual Teaching and Education

Augmented reality (AR) and virtual reality (VR) as virtual teaching methods are also starting to offer innovative ways to educate the medical professionals of tomorrow.<sup>47,48</sup> By providing access to real-world simulations and hands-on training, these methods can help improve learner engagement and outcomes.<sup>49</sup> By allowing users to interact within a virtual 3-dimensional, computer-generated environment within seemingly real or physical ways, such immersive technologies can provide an entirely simulated environment (eg, VR) or one that is superimposed atop our real surroundings (eg, AR) to enhance engagement. In the context of medical education, virtual education offers a highly engaging and interactive platform in which learners can explore anatomical structures, practice surgical procedures, and simulate clinical scenarios. In the field of medicine, virtual teaching can enhance the quality of education and prepare students for the challenges of the real world.<sup>50</sup> These virtual education technologies typically involve the use of head-mounted devices or virtual devices equipped with specialized equipment to track head, eye, and hand movements to provide sensory feedback, enhancing the educational immersion.

Virtual education enhances learning outcomes by providing a hands-on, simulated learning environment. Pathology and medical education for that matter traditionally rely on textbooks, slide review, instrument evaluations, and cadaveric dissections. Virtual education offers a transformative experience by allowing students to visualize complex anatomical structures in 3 dimensions, manipulate virtual specimens, and observe dynamic physiological processes in real time. This immersive learning approach fosters deeper understanding and retention of knowledge compared with traditional methods. Virtual education enables personalized learning experiences in which students can proceed at their own pace, repeat procedures or examinations ad infinitum, and receive feedback using novel methods. Integrating AI-ML with virtual education represents a transformative leap in medical and health care training.<sup>51</sup> AI enhances personalized learning by adapting content and pace to individual student needs, optimizing comprehension and retention. AI-driven analytics provide insights into student progress and learning gaps, informing educators on effective instructional strategies and curriculum development. This interactivity promotes active engagement and encourages critical thinking and problem-solving skills essential for diagnostic reasoning for most fields in medicine.

Furthermore, the integration of AI-ML into virtual teaching and education is exemplified through intelligent tutoring systems (ITS), virtual assistants, interactive learning (eg, gamification), smart content creation, and tailored learning experiences. ITS leverage AI to offer personalized educational experiences that adapt to individual learners’ needs. Unlike traditional educational software, ITS dynamically adjusts to the learner’s progress and learning style, providing tailored feedback and instruction. These  systems use AI-ML algorithms to analyze user interactions in real time, allowing for immediate adjustments in content difficulty and pedagogic approach. For instance, in medical education, ITS can guide students through diagnostic problem solving by analyzing their choices, offering hints or alternative strategies, and adapting complexity to their competency levels. This approach simulates the benefits of one-on-one tutoring, making it particularly effective for complex or highly specialized training such as pathology workflows. In addition, the integration of AR with ITS can enhance learning by creating interactive, immersive environments that make abstract concepts more tangible and engaging.

Virtual assistants in education are another AI-driven application designed to enhance learning experiences. These systems use NLP and ML algorithms to perform a range of tasks based on user commands and queries. For example, AI virtual assistants could simulate patient interviews in medical training, offering real-time feedback and responses based on user inputs. These assistants continuously learn from user interactions, improving their ability to perform tasks and guide learners effectively. When combined with VR or AR, virtual assistants can interact with the physical environment in novel ways, such as displaying virtual controls or information overlays. In educational contexts, this might include guiding medical students through anatomical dissections or offering step-by-step assistance during simulated surgical procedures. Such tools are particularly valuable in clinical education, in which students must practice hands-on skills in a controlled, risk-free virtual environment. Interactive learning, including gamification, incorporates game design elements into educational contexts to increase engagement and motivation. AI-ML plays a pivotal role in gamification by analyzing learner performance data to dynamically tailor challenges, content, and feedback. For example, in medical education, an AI-powered gamified platform could adapt the difficulty of diagnostic case simulations based on the learner’s performance in previous sessions. This dynamic adjustment ensures that gamified elements remain engaging, appropriately challenging, and educationally effective. VR and/or AR can amplify the effects of interactive learning by creating immersive scenarios in which learners interact with the educational content superimposed on the real world. For instance, VR environments could allow pathology trainees to examine 3D histologic slides in an interactive, engaging manner that mimics real-world diagnostic tasks. Smart content creation involves the use of AI-ML to generate and curate materials customized to meet specific learner needs and contexts. AI-ML algorithms can analyze curriculum requirements and student performance data to create adaptive learning materials, such as quizzes, case studies, and interactive modules. For example, in pathology education, AI-ML systems could dynamically generate case studies featuring rare diagnostic conditions tailored to the needs of specific learners. These tools also facilitate content personalization and updates, ensuring that materials remain relevant and effective. When combined with virtual environments, AI-powered smart content creation can transform traditional materials into interactive 3D models or simulations, such as visualizing cellular mechanisms or creating virtual tissue samples for detailed analysis. AI-ML uniquely enhances virtual education by enabling personalization, adaptive learning, and immersive experiences. Unlike traditional digital platforms or VR/AR tools used in isolation, AI-ML actively drives content generation, interaction analysis, and user-specific adjustments, making learning more efficient and impactful. By integrating these intelligent systems, virtual education frameworks are not only engaging but also responsive to learners’ individual needs,

---

bridging the gap between theoretical knowledge and practical application.

Education simulations provide a safe and controlled environment for practicing diagnostic procedures and surgical techniques without the constraints of patient safety or availability of cadaveric specimens (eg, autopsy). In pathology, virtual education can replicate the examination of tissue samples under different conditions (eg, varying magnifications and stains) and simulate the interpretation of histopathological findings. Virtual fine needle aspiration clinics can simulate patient experiences, needle localization procedures, and even smearing of glass slides using virtual handheld device tracking, in addition to review of the WSI. Laboratory medicine training benefits from virtual education by offering virtual laboratories in which students can perform diagnostic tests, analyze results, and troubleshoot technical issues in a realistic setting. This virtual "hands-on" training prepares students for real-world challenges and improves procedural proficiency before clinical rotations or professional practice. Furthermore, virtual education facilitates interdisciplinary learning by simulating collaborative health care scenarios in which pathologists, clinicians, and laboratory technologists work together to diagnose and manage patient cases. This teamwork simulation enhances communication skills and promotes a holistic understanding of patient care pathways. By leveraging immersive technology, educators can enhance learning outcomes, engage students in interactive learning experiences, and provide access to realistic simulations that mirror clinical practice. As virtual education continues to evolve, its integration into curricula holds the potential to revolutionize how future health care professionals are trained, ensuring that they are well prepared to meet the complex demands of medical discipline in an increasingly technology-driven health care environment.

# Artificial Intelligence in Medical Research

Future ML platforms have potential to reform the landscape of scientific research in health care. One of the most profound impacts of AI-ML is the ability to rapidly process and analyze vast data sets, and identify complex patterns and correlations. This will surpass existing data discovery methodologies, particularly in subspecialties such as genomics, medical imaging, and population health. In the domain of genomics, AI-ML tools can examine genetic read data to pinpoint potential biomarkers for diseases, a task crucial for the development of targeted therapies and precision medicine. AI-ML has identified novel biomarkers in radiomics and pathomics and expanded molecular biomarker discovery beyond genomics to include transcriptomics and epigenomics.<sup>52-54</sup> This can enable more robust and comprehensive insights into disease mechanisms and public health trends.<sup>55,56</sup> More recently, digital biobanks have emerged to enable the storage, management, and analysis of vast amounts of biological data. AI-ML enhances the efficiency of digital biobanks by improving data integration and facilitating advanced queries and analyses in addition to being able to process and identify trends within digital data. This development supports more effective health care research by providing researchers with robust, well-organized, and prescreened data. Recent literature explores the role of AI in optimizing digital biobanks, generating synthetic data to support digital research, and emphasizing its impact on data accessibility and research productivity.<sup>57-60</sup> Digital biobanks can support clinical trials by providing authentic or synthetic data. AI-ML platforms enable ingesting and leveraging these data to integrate into clinical trial design, patient selection, trial execution, and result analysis. AI-ML facilitates

participant recruitment by identifying suitable candidates based on comprehensive criteria. Moreover, adaptive trial designs enabled by AI allow for real-time modifications to trial protocols, optimizing outcomes and reducing costs. Simulation of clinical trials with digital twins provides advanced simulations that replicate real-world systems, processes, or data using AI-ML. They create dynamic, virtual models that accurately reflect the current state and behavior of their physical counterparts. Peshkova et al<sup>61</sup> display this concept within the pathology domain creating a database of colorectal carcinoma. Creating intricate digital models using biobank data from patient data can enhance the development of advanced diagnostic tools. The use of AI-ML in epidemiology is also becoming increasingly prominent, particularly in predicting disease outbreaks and assessing public health risks. AI models can forecast the spread of diseases, as demonstrated in studies related to COVID-19, providing crucial insights for public health planning and response (Fig. 4).<sup>62,63</sup>

In the field of drug discovery, AI algorithms can accelerate drug development by identifying potential drug targets and repurposing existing drugs for new therapeutic uses. Using AI-ML platforms to analyze large data sets can help predict patient responses to various treatment regimens and select the best patient treatment strategies. Predictive modeling further enhances this by simulating drug interactions with biological systems, thereby forecasting efficacy and side effects before clinical trials. The potential of AI in de novo drug design and vaccine development shows how ML can predict the effectiveness of new drugs and optimize their development process.<sup>64-67</sup> In tandem, personalized medicine is another significant beneficiary of AI-ML platforms. AI-ML enables the analysis of genetic, clinical, and lifestyle data to stratify patients into subgroups that can benefit from specific treatments. This approach enhances treatment efficacy by tailoring interventions to individual patient profiles. AI-ML platforms are also making strides in spatial biology, tumor microenvironment analysis, and multiplexing technologies. The ability to process and interpret complex spatial and multiomic data, providing insights into how various biological components interact within tissues and tumors, will be critical to study their impact on patients. Fu et al<sup>68</sup> illustrate pathology modeling to enhance the resolution of spatial biology and tumor microenvironment for predicting patient prognosis and therapeutic response.

In scholarly research, large language models have emerged as valuable tools for literature search, summarization, and synthesis. These models assist researchers in efficiently answering complex research questions and generating outlines or synopses of publications. By analyzing extensive bodies of literature, large language models facilitate the identification of key findings and gaps in knowledge, thereby supporting more informed and impactful research. Overall, the future of ML platforms promises to enhance various aspects of scientific research and health care, from data analysis and drug discovery to personalized medicine and clinical trials. As these technologies continue to evolve, they will undoubtedly drive significant advancements in our understanding and management of complex biological and medical challenges.

# Regulatory Considerations

Regulatory bodies play an important role in ensuring the safe and effective use of AI-ML platforms. This section will focus on regulatory considerations specific to such platforms; however, there is a more in-depth review on regulatory aspects of AI-ML. Establishing comprehensive standards and guidelines for the development, validation, and deployment of AI systems is

---

**Virtual Reality (VR)**
* Learner is completely immersed in a self-contained digital environment.
* Often the goggles are fully closed off with opaque plastic.
* Remote controls usually needed to control virtual hands due to total enclosure.

![Illustration of VR and AR headsets](page_10_image_5_v2.jpg)

**Augmented Reality (AR)**
* Learner interacts with holograms superimposed into the real environment.
* Often the goggles are separated from the world by a piece of dark transparent glass/plastic with cameras/sensors. Can also operate through phone camera and screen.
* Possibility to use one's own hands as controls

**VR Cadaveric Dissection**
![VR Cadaveric Dissection simulation](page_10_image_2_v2.jpg)

**Examples of Each**
| VR                  | AR                   |
| ------------------- | -------------------- |
| - Meta Oculus Quest | - Apple Vision Pro   |
| - HTC Vive          | - Microsoft HoloLens |
| - Valve Index       | - XReal Air 2        |

**AR Cadaveric Dissection**
![AR Cadaveric Dissection simulation](page_10_image_4_v2.jpg)

**VR Histology Learning**
![VR Histology Learning simulation](page_10_image_3_v2.jpg)

By using VR/AR to digitize learning, medical trainees' progress can be transmitted to and processed by a central AI, which can then provide detailed progress reports to instructors.

![Diagram showing data transmission from VR/AR to AI](page_10_image_6_v2.jpg)

**AR Histology Learning**
![AR Histology Learning simulation](page_10_image_1_v2.jpg)

Figure 4.
Virtual education in pathology.

essential. This includes setting criteria for data quality, model performance, and clinical validation. Regulatory agencies should also promote ongoing surveillance and postmarket monitoring of AI-ML platforms to identify and address potential issues in real-world settings. Given the global nature of health care, international collaboration is necessary to harmonize ethical and regulatory standards for AI-ML platforms. Collaborative efforts can facilitate the sharing of best practices, promote interoperability, and ensure that AI systems meet high standards of safety and efficacy across different regions. International bodies, such as the World Health Organization and other regulatory organizations, can play a key role in fostering such collaboration.

There is a growing need for clarity on accountability and liability, especially as AI systems become more integrated into clinical decision-making. Future legislation may address these issues, balancing the responsibilities between AI developers, health care providers, and institutions. Current regulations in health care firmly position (human) health care professionals as being responsible for the care of a patient. This is likely to be maintained for the near future as the risks associated with AI, and the reasons why AI sometimes fails when they do, remain unclear. This creates an increasingly precarious situation in which health care providers may be ill-equipped to bear the responsibility for mistakes made by AI systems. Laboratory and medical directors in particular will be burdened with managing the risk/benefit equation within their health care organizations. Future legislation that speaks to tort reform and changing standards of care in a setting of prevalent AI deployments seems inevitable. Some publications in legal academia have begun to address this issue, but much more thought leadership and consensus building are required.<sup>72,73</sup>

Future deployments of AI in health care will likely be subject to evolving regulations as risks to patients become clearer. Determining accountability and liability in cases in which AI-ML systems contribute to clinical decisions is complex. Clear guidelines are needed to delineate the responsibilities of AI developers, health care providers, and institutions. In the United States, various agencies such as the Food and Drug Administration, the Centers for Medicare & Medicaid Services, US Department of Health and Human Services, Centers for Disease Control and Prevention, and Office of the National Coordinator for Health Information Technology oversee different aspects of health care regulation and will likely influence future AI regulations. The Food and Drug Administration, although not yet having specific AI regulations, has proposed frameworks for AI/ML-based medical devices, acknowledging the need for adaptive regulatory approaches owing to the evolving nature of AI technologies.<sup>69-71</sup> Nongovernmental organizations, such as the Coalition for Healthcare AI, are also contributing to the development of safety standards and evaluation criteria for AI in health care.

**Conclusion**

The integration of AI-ML platforms into health care systems represents a transformative shift toward enhanced diagnostic accuracy, streamlined clinical workflows, and personalized medicine. Current trends include the adoption of ML-Ops for efficient model management in the clinical environment, the development and deployment of multimodal AI to leverage diverse data sources, multiagent frameworks, and advancements toward AGI to revolutionize medical decision-making. As AI systems continue to demonstrate their potential in medicine, a sustained need for substantial investment and research within this new arena is becoming critical. This includes fostering interdisciplinary

As AI technologies continuously evolve, current regulations, which typically apply to static medical devices, may need to adapt.

---

collaborations among clinicians, data scientists, and engineers to develop robust AI algorithms tailored for health care settings.

Looking forward, the impact of AI-ML platforms in medicine is poised to be profound. In practice, AI-driven diagnostics will improve accuracy and productivity, ultimately leading to improved patient outcomes and reduced health care costs. In research, AI will facilitate deeper insights into disease mechanisms and accelerate data-drive patient management through predictive modeling and personalized medicine approaches. In education, AI-ML technologies, coupled with virtual teaching methods, will transform how the next generation of pathologists and other medical professionals are trained. Access to realistic simulations and interactive learning environments will enhance understanding and proficiency, preparing students to tackle complex challenges in real clinical settings.

By embracing these technologies thoughtfully and proactively, health care systems can advance toward more effective, patient-centered care paradigms that harness the full potential of AI to improve human health globally. Continued collaboration, investment, and ethical and regulatory consideration are needed to navigate this transformative journey in health care innovation.

### Acknowledgments

This article series is part of the educational effort of Computational Pathology and AI Center of Excellence at the university of Pittsburgh.

### Author Contributions

M.G.H., L.P., R.D., J.H.H., M.D., and H.H.R. contributed to manuscript text contents. All images/figures were constructed by J. P.

### Data Availability

Not applicable.

### Funding

Not applicable.

### Declaration of Competing Interest

L.P. is a consultant for Hamamatsu, AiXMed, and NTP, serves on the advisory board for Ibex, and is a co-owner of Placenta AI and Lean AP. H.H.R. is a creator of MILO and also a coinventor of STNG.

### Ethics Approval and Consent to Participate

Not applicable.

### Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

During the preparation of this work the authors used DALL-E via ChatGPT-4o, as well as Adobe Express Online, in order to generate de novo artworks for the creation of certain figures or parts of figures. Additionally, a large language model was used to generate parts of the initial manuscript outline, but the text content of this manuscript is all human generated and did not include generative AI outputs.

### References

1. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56. https://doi.org/10.1038/s41591-018-0300-7

2. Jiang F, Jiang Y, Zhi H, et al. Artificial intelligence in healthcare: past, present and future. Stroke Vasc Neurol. 2017;2(4):230-243. https://doi.org/10.1136/svn-2017-000101

3. Yu KH, Beam AL, Kohane IS. Artificial intelligence in healthcare. Nat Biomed Eng. 2018;2(10):719-731. https://doi.org/10.1038/s41551-018-0305-z

4. Gore JC. Artificial intelligence in medical imaging. Magn Reson Imaging. 2020;68:A1-A4. https://doi.org/10.1016/j.mri.2019.12.006

5. Barragan-Montero A, Javaid U, Valdes G, et al. Artificial intelligence and machine learning for medical imaging: a technology review. Phys Med. 2021;83:242-256. https://doi.org/10.1016/j.ejmp.2021.04.016

6. Campanella G, Hanna MG, Geneslaw L, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat Med. 2019;25(8):1301-1309. https://doi.org/10.1038/s41591-019-0508-1

7. Rajpara SM, Botello AP, Townend J, Ormerod AD. Systematic review of dermoscopy and digital dermoscopy/ artificial intelligence for the diagnosis of melanoma. Br J Dermatol. 2009;161(3):591-604. https://doi.org/10.1111/j.1365-2133.2009.09093.x

8. Ting DSW, Pasquale LR, Peng L, et al. Artificial intelligence and deep learning in ophthalmology. Br J Ophthalmol. 2019;103(2):167-175. https://doi.org/10.1136/bjophthalmol-2018-313173

9. Stålhammar G, Fuentes Martinez N, Lippert M, et al. Digital image analysis outperforms manual biomarker assessment in breast cancer. Mod Pathol. 2016;29(4):318-329. https://doi.org/10.1038/modpathol.2016.34

10. Cornish TC. Clinical application of image analysis in pathology. Adv Anat Pathol. 2020;27(4):227-235. https://doi.org/10.1097/PAP.0000000000000263

11. Holten-Rossing H, Talman MM, Jylling AMB, Laenkholm AV, Kristensson M, Vainer B. Application of automated image analysis reduces the workload of manual screening of sentinel lymph node biopsies in breast cancer. Histopathology. 2017;71(6):866-873. https://doi.org/10.1111/his.13305

12. Volynskaya Z, Mete O, Pakbaz S, Al-Ghamdi D, Asa SL. Ki67 quantitative interpretation: insights using image analysis. J Pathol Inform. 2019;10:8. https://doi.org/10.4103/jpi.jpi_76_18

13. Gil J, Wu HS. Applications of image analysis to anatomic pathology: realities and promises. Cancer Invest. 2003;21(6):950-959. https://doi.org/10.1081/cnv-120025097

14. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. N Engl J Med. 2019;381(20):1909-1917. https://doi.org/10.1056/NEJMoa1901183

15. Nahavandi D, Alizadehsani R, Khosravi A, Acharya UR. Application of artificial intelligence in wearable devices: opportunities and challenges. Comput Methods Programs Biomed. 2022;213:106541. https://doi.org/10.1016/j.cmpb.2021.106541

16. Lubitz SA, Faranesh AZ, Selvaggi C, et al. Detection of atrial fibrillation in a large population using wearable devices: the Fitbit Heart Study. Circulation. 2022;146(19):1415-1424. https://doi.org/10.1161/CIRCULATIONAHA.122.060291

17. Hijazi H, Abu Talib M, Hasasneh A, Bou Nassif A, Ahmed N, Nasir Q. Wearable devices, smartphones, and interpretable artificial intelligence in combating COVID-19. Sensors (Basel). 2021;21(24):8424. https://doi.org/10.3390/s21248424

18. Mäkynen M, Ng GA, Li X, Schlindwein FS. Wearable devices combined with artificial intelligence-a future technology for atrial fibrillation detection? Sensors (Basel). 2022;22(22):8588. https://doi.org/10.3390/s22228588

19. Wang WH, Hsu WS. Integrating artificial intelligence and wearable IoT system in long-term care environments. Sensors (Basel). 2023;23(13):5913. https://doi.org/10.3390/s23135913

20. Hossain E, Rana R, Higgins N, et al. Natural language processing in electronic health records in relation to healthcare decision-making: a systematic review. Comput Biol Med. 2023;155:106649. https://doi.org/10.1016/j.compbiomed.2023.106649

21. Cui M, Zhang DY. Artificial intelligence and computational pathology. Lab Invest. 2021;101(4):412-422. https://doi.org/10.1038/s41374-020-00514-0

22. Huyen C. Designing Machine Learning Systems. O'Reilly Media Inc; 2022. Accessed June 23, 2024. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/

23. Sculley D, Holt G, Golovin D, et al. Hidden technical debt in machine learning systems. In: Proceedings of the 28th International Conference on Neural Information Processing Systems-Volume 2. MIT Press; 2015:2503-2511. NIPS'15.

24. Abramoff MD, Whitestone N, Patnaik JL, et al. Autonomous artificial intelligence increases real-world specialist clinic productivity in a cluster-randomized trial. NPJ Digit Med. 2023;6(1):184. https://doi.org/10.1038/s41746-023-00931-7

25. Khattak FK, Subasri V, Krishnan A, et al. MLHOps: machine learning for healthcare operations. Published online May 3, 2023. Accessed June 23, 2024. http://arxiv.org/abs/2305.02474

26. Davis SE, Greevy RA Jr, Lasko TA, Walsh CG, Matheny ME. Detection of calibration drift in clinical prediction models to inform model updating. J Biomed Inform. 2020;112:103611. https://doi.org/10.1016/j.jbi.2020.103611

---

27. Davis SE, Greevy Jr RA, Fonnesbeck C, Lasko TA, Walsh CG, Matheny ME. A nonparametric updating method to correct clinical prediction model drift. *J Am Med Inform Assoc.* 2019;26(12):1448–1457. https://doi.org/10.1093/jamia/ocz127

28. Dan J, Seeuws N. Enhancing reproducibility of machine learning-based studies in clinical journals through model cards. *Dev Med Child Neurol.* 2024;66(2):144–145. https://doi.org/10.1111/dmcn.15785

29. Amith MT, Cui L, Zhi D, et al. Toward a standard formal semantic representation of the model card report. *BMC Bioinformatics.* 2022;23(suppl 6):281. https://doi.org/10.1186/s12859-022-04797-6

30. Amith MT, Cui L, Roberts K, Tao C. Application of an ontology for model cards to generate computable artifacts for linking machine learning information from biomedical research. *Proc Int World Wide Web Conf.* 2023;2023(Companion):820–825. https://doi.org/10.1145/3543873.3587601

31. Morik KJ, Kotthaus H, Fischer R, et al. Yes we care!-Certification for machine learning methods through the care label framework. *Front Artif Intell.* 2022;5:975029. https://doi.org/10.3389/frai.2022.975029

32. Krumm N. Organizational and technical security considerations for laboratory cloud computing. *J Appl Lab Med.* 2023;8(1):180–193. https://doi.org/10.1093/jalm/jfac118

33. Sacco A, Esposito F, Marchetto G, Kolar G, Schwetye K. On edge computing for remote pathology consultations and computations. *IEEE J Biomed Health Inform.* 2020;24(9):2523–2534. https://doi.org/10.1109/JBHI.2020.3007661

34. Yousif M, Balis UGJ, Parwani AV, Pantanowitz L. Commentary: leveraging edge computing technology for digital pathology. *J Pathol Inform.* 2021;12:12. https://doi.org/10.4103/jpi.jpi_112_20

35. Clinical Laboratory Improvement Amendments of 1988 (CLIA) Title 42: The Public Health and Welfare. Subpart 2: Clinical Laboratories (42 U.S.C. 263a). Accessed April 18, 2022. https://www.govinfo.gov/content/pkg/USCODE-2011-title42/pdf/USCODE-2011-title42-chap6A-subchapII-partF-subpart2-sec263a.pdf

36. Hanna MG, Olson NH, Zarella M, et al. Recommendations for performance evaluation of machine learning in pathology: a concept paper from the College of American Pathologists. *Arch Pathol Lab Med.* 2023;148(10):e335–e361. https://doi.org/10.5858/arpa.2023-0042-CP

37. Ngiam KY, Khor IW. Big data and machine learning algorithms for health-care delivery. *Lancet Oncol.* 2019;20(5):e262–e273. https://doi.org/10.1016/S1470-2045(19)30149-4

38. Chen RJ, Lu MY, Williamson DFK, et al. Pan-cancer integrative histologic-genomic analysis via multimodal deep learning. *Cancer Cell.* 2022;40(8):865–878.e6. https://doi.org/10.1016/j.ccell.2022.07.004

39. Lipkova J, Chen RJ, Chen B, et al. Artificial intelligence for multimodal data integration in oncology. *Cancer Cell.* 2022;40(10):1095–1110. https://doi.org/10.1016/j.ccell.2022.09.012

40. Xia Y, Ji Z, Krylov A, Chang H, Cai W. Machine learning in multimodal medical imaging. *Biomed Res Int.* 2017;2017:1278329. https://doi.org/10.1155/2017/1278329

41. Huang Z, Tanaka F. MSPM: a modularized and scalable multi-agent reinforcement learning-based system for financial portfolio management. *PLoS One.* 2022;17(2):e0263689. https://doi.org/10.1371/journal.pone.0263689

42. Montagna S, Castro Silva D, Henriques Abreu P, Ito M, Schumacher MI, Vargiu E. Autonomous agents and multi-agent systems applied in healthcare. *Artif Intell Med.* 2019;96:142–144. https://doi.org/10.1016/j.artmed.2019.02.007

43. Bennai MT, Guessoum Z, Mazouzi S, Cormier S, Mezghiche M. Multi-agent medical image segmentation: a survey. *Comput Methods Programs Biomed.* 2023;232:107444. https://doi.org/10.1016/j.cmpb.2023.107444

44. Mutlag AA, Khanapi Abd Ghani M, Mohammed MA, et al. MAFC: multi-agent fog computing model for healthcare critical tasks management. *Sensors (Basel).* 2020;20(7):1853. https://doi.org/10.3390/s20071853

45. Stone P, Kaminka G, Kraus S, Rosenschein J. Ad hoc autonomous agent teams: collaboration without pre-coordination. *Proceedings of the AAAI Conference on Artificial Intelligence.* 2010;24(1):1504–1509. https://doi.org/10.1609/aaai.v24i1.7529

46. Goertzel B, Pennachin C, eds. *Artificial General Intelligence.* Springer; 2007. https://doi.org/10.1007/978-3-540-68677-4

47. Huang HM, Liaw SS. An analysis of learners' intentions toward virtual reality learning based on constructivist and technology acceptance approaches. *Int Rev Res Open Distrib Learn.* 2018;19(1). https://doi.org/10.19173/irrodl.v19i1.2503

48. Hassell LA, Absar SF, Chauhan C, et al. Pathology education powered by virtual and digital transformation: now and the future. *Arch Pathol Lab Med.* 2022;147(4):474–491. https://doi.org/10.5858/arpa.2021-0473-RA

49. Campos E, Hidrogo I, Zavala G. Impact of virtual reality use on the teaching and learning of vectors. *Front Educ.* 2022;7. https://doi.org/10.3389/feduc.2022.965640

50. Hanna MG, Ahmed I, Nine J, Prajapati S, Pantanowitz L. Augmented reality technology using Microsoft Hololens in anatomic pathology. *Arch Pathol Lab Med.* 2018;142(5):638–644. https://doi.org/10.5858/arpa.2017-0189-OA

51. O'Sullivan S, Holzinger A, Zatloukal K, Saldiva P, Sajid MI, Wichmann D. Machine learning enhanced virtual autopsy. *Autops Case Rep.* 2017;7(4):3–7. https://doi.org/10.4322/acr.2017.037

52. Prelaj A, Miskovic V, Zanitti M, et al. Artificial intelligence for predictive biomarker discovery in immuno-oncology: a systematic review. *Ann Oncol.* 2024;35(1):29–65. https://doi.org/10.1016/j.annonc.2023.10.125

53. Mikdadi D, O'Connell KA, Meacham PJ, et al. Applications of artificial intelligence (AI) in ovarian cancer, pancreatic cancer, and image biomarker discovery. *Cancer Biomark.* 2022;33(2):173–184. https://doi.org/10.3233/CBM-210301

54. Çalışkan M, Tazaki K. AI/ML advances in non-small cell lung cancer biomarker discovery. *Front Oncol.* 2023;13:1260374. https://doi.org/10.3389/fonc.2023.1260374

55. Rasmusson A, Zilenaite D, Nestarenkaite A, et al. Immunogradient indicators for antitumor response assessment by automated tumor-stroma interface zone detection. *Am J Pathol.* 2020;190(6):1309–1322. https://doi.org/10.1016/j.ajpath.2020.01.018

56. Chi J, Shu J, Li M, et al. Artificial intelligence in metabolomics: a current review. *Trends Analyt Chem.* 2024;178:117852. https://doi.org/10.1016/j.trac.2024.117852

57. Bonizzi G, Zattoni L, Fusco N. Biobanking in the digital pathology era. *Oncol Res.* 2022;29(4):229–233. https://doi.org/10.32604/or.2022.024892

58. Brancato V, Esposito G, Coppola L, et al. Standardizing digital biobanks: integrating imaging, genomic, and clinical data for precision medicine. *J Transl Med.* 2024;22(1):136. https://doi.org/10.1186/s12967-024-04891-8

59. Frascarelli C, Bonizzi G, Musico CR, et al. Revolutionizing cancer research: the impact of artificial intelligence in digital biobanking. *J Pers Med.* 2023;13(9):1390. https://doi.org/10.3390/jpm13091390

60. Pantanowitz J, Manko CD, Pantanowitz L, Rashidi HH. Synthetic data and its utility in pathology and laboratory medicine. *Lab Invest.* 2024;104(8):102095. https://doi.org/10.1016/j.labinv.2024.102095

61. Peshkova M, Yumasheva V, Rudenko E, Kretova N, Timashev P, Demura T. Digital twin concept: healthcare, education, research. *J Pathol Inform.* 2023;14:100313. https://doi.org/10.1016/j.jpi.2023.100313

62. Ankolekar A, Eppings L, Bottari F, et al. Using artificial intelligence and predictive modelling to enable learning healthcare systems (LHS) for pandemic preparedness. *Comput Struct Biotechnol J.* 2024;24:412–419. https://doi.org/10.1016/j.csbj.2024.05.014

63. Demirbaga U, Kaur N, Aujla GS. Uncovering hidden and complex relations of pandemic dynamics using an AI driven system. *Sci Rep.* 2024;14(1):15433. https://doi.org/10.1038/s41598-024-65845-0

64. Gholap AD, Uddin MJ, Faiyazuddin M, Omri A, Gowri S, Khalid M. Advances in artificial intelligence for drug delivery and development: a comprehensive review. *Comput Biol Med.* 2024;178:108702. https://doi.org/10.1016/j.compbiomed.2024.108702

65. Jimeno A, Moore KN, Gordon M, et al. A first-in-human phase 1a study of the bispecific anti-DLL4/anti-VEGF antibody navicixizumab (OMP-305B83) in patients with previously treated solid tumors. *Invest New Drugs.* 2019;37(3):461–472. https://doi.org/10.1007/s10637-018-0665-y

66. Visan AI, Negut I. Integrating artificial intelligence for drug discovery in the context of revolutionizing drug delivery. *Life (Basel).* 2024;14(2):233. https://doi.org/10.3390/life14020233

67. Singh S, Kumar R, Payra S, Singh SK. Artificial intelligence and machine learning in pharmacological research: bridging the gap between data and drug discovery. *Cureus.* 2023;15(8):e44359. https://doi.org/10.7759/cureus.44359

68. Fu X, Sahai E, Wilkins A. Application of digital pathology-based advanced analytics of tumour microenvironment organisation to predict prognosis and therapeutic response. *J Pathol.* 2023;260(5):578–591. https://doi.org/10.1002/path.6153

69. IMDRF Software as a Medical Device (SaMD) Working Group. "Software as a Medical Device": possible framework for risk categorization and corresponding considerations. 30. Published online 2014. Accessed June 22, 2024. https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-140918-samd-framework-risk-categorization-141013.pdf

70. US Food and Drug Administration. Proposed regulatory framework for modifications to artificial intelligence/machine learning (AI/ML)-based software as a medical device (SAMD). US FDA Artificial Intelligence and Machine Learning Discussion Paper. FDA. Published online April 2, 2019. Accessed December 30, 2021. https://www.fda.gov/media/122535/download

71. US Food and Drug Administration. Artificial intelligence and machine learning in software as a medical device. FDA. Published online September 22, 2021. Accessed December 30, 2021. https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device

72. Weil G. Tort law as a tool for mitigating catastrophic risk from artificial Intelligence. *SSRN Journal.* Published online January 13, 2024. https://doi.org/10.2139/ssrn.4694006

73. Allain JS. From jeopardy! to jaundice: the medical liability implications of Dr. Watson and other artificial intelligence systems. *LA Law Rev.* 2013;73:4.