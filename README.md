# README.md

## Project: Patient Trial Matching with Clinical Data and LLM Integration

### Overview

This repository contains a project aimed at matching patient profiles with clinical trials using both structured and unstructured data. It leverages rule-based logic along with NER using SciSpacy models and [RAG] Large Language Models (LLMs) to improve the matching process, particularly for interpreting complex trial eligibility criteria and patient medical conditions.

The repository is structured with modular functions, including handling trial criteria checks, LLM integration, patient data processing, and clinical trial data handling. The primary submission notebook demonstrates the end-to-end process.

### Setup

1. Clone the repository:

```bash
git clone https://github.com/arun98aol/patient-trial-match-scispacy-rag.git
cd patient-trial-match-scispacy-rag
```

Follow the notebook for next steps - **Patient Trial Match [Submission].ipynb**
 - Python 3.11.9, Windows.
 - Refer functions section below to change LLM APIs.
 - This notebook contains the takeaways and challenges faced.
 - Trials data is not uploaded to this repo, so, use reload=True to fetch again or download a sample dataset online. (takes hours to get all, so set n=1000)
 - Outputs (JSON and CSV) in data/output folder

### Folder Structure (Here's how it should look for everything to run great)

```bash
.
├── data
│   ├── annotated_synthea/           # Processed synthetic dataset with UMLS codes.
│   ├── example_patient_logs/        # Example patient logs.
│   ├── output/                      # Task output folder.
│   ├── patient/                     # Processed patient files.
│   ├── trials/                      # Processed trials files. Should contain df_trials_processed.csv
│   ├── LLM_PromptExample1.txt       # Brainstorming with large language models (example 1).
│   ├── LLM_PromptExample2.txt       # Brainstorming with large language models (example 2).
│   └── umls_codes.csv               # Relevant UMLS codes for trials and patients (definitions and aliases), generated using SciSpacy models.
├── functions
│   ├── __pycache__/                 # Python cache files.
│   ├── criteria_checks.py           # Contains all the basic criteria checks.
│   ├── llm_apis.py                  # Handles API calls to large language models.
│   ├── patient_trial_matcher.py     # Main matching algorithm for patient-trial.
│   ├── patients.py                  # Patient data processing scripts.
│   ├── scispacy_models.py           # SciSpacy model processing.
│   └── trials.py                    # Trial data processing scripts.
├── synthea_100/                     # Original downloaded sample dataset of patients.
├── .env                             # Important! Contains API keys and environment variables.
├── .gitignore                       # Standard Git ignore file for excluding certain files from the repository.
├── 00. Data Collection & Research.ipynb   # Notebook for data collection and research.
├── 01. Data Exploration.ipynb       # Notebook for initial data exploration.
├── 02. Setting Up Basic Match.ipynb # Notebook for setting up a basic match algorithm.
├── LICENSE                          # License for the project.
├── Patient Trial Match [Submission].ipynb   # Main submission notebook for patient-trial matching.
├── README.md                        # README file with project overview.
└── requirements.txt                 # File listing Python dependencies for the project.
```

### Functions

- **llm_apis.py:**
  - This module contains functions that call LLM APIs (e.g., OpenAI's GPT or Mistral API) to help interpret trial criteria when they are not easily interpretable using traditional logic. **The call_openai_api function decides which API is used (Azure/OpenAI) incase you need to modify that. Additionally, the 'patient_trial_matcher.py' file defines which api is used OpenAI or Mistral incase you need to change that.**

- **patient_trial_matcher.py:**
  - The main logic for matching patients to clinical trials. This function integrates rule-based matching, LLM-assisted checks, and generates output in JSON format.

- **patients.py:**
  - Contains helper functions for loading, processing, and transforming patient data.

- **scispacy_models.py:**
  - Utilizes SciSpacy and UMLS models to extract useful entities from clinical notes, patient profiles, and trial descriptions.

- **trials.py:**
  - Functions for handling trial data, including extracting criteria and preparing them for matching.

### Data Overview

- **annotated_synthea/**:
  - Contains annotated synthetic patient data generated using Synthea, with added labels and UMLS codes for conditions and medications.

- **example_patient_logs/**:
  - Example logs generated from patient-trial matching processes, useful for debugging and analyzing the pipeline’s behavior.

- **output/**:
  - This folder contains JSON outputs and logs generated during the patient-trial matching process.

- **umls_codes.csv:**
  - A CSV file that stores extracted UMLS codes for conditions and medications used in matching patients to clinical trials.
