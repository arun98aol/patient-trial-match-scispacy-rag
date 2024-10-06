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
 - I used Python 3.11.9, Windows.
 - Refer functions section below to change LLM APIs.
 - This notebook contains the takeaways and challenges faced.
 - Trials data is not uploaded to this repo, so, use reload=True to fetch again or download a sample dataset online.

### Folder Structure

```bash
.
├── .venv/                          # Virtual environment for dependencies (not shared)
├── data/                           # Data folder
│   ├── annotated_synthea/          # Annotated Synthea patient data
│   ├── example_patient_logs/       # Logs generated from matching patients with trials
│   ├── output/                     # JSON and log outputs for matched results
│   ├── patient/                    # Folder containing patient-related data
│   ├── trials/                     # Folder containing clinical trial data
│   ├── umls_codes.csv              # UMLS codes extracted for patient-trial matching
├── functions/                      # Python scripts for core functionalities
│   ├── criteria_checks.py          # Functions to check inclusion/exclusion criteria
│   ├── llm_apis.py                 # Functions handling LLM API calls (e.g., OpenAI/Mistral)
│   ├── patient_trial_matcher.py    # Main logic for matching patient profiles to trials
│   ├── patients.py                 # Functions for processing patient data
│   ├── scispacy_models.py          # Functions for integrating SciSpacy and UMLS models
│   ├── trials.py                   # Functions for processing clinical trial data
├── synthea_100/                    # Original synthetic patient data from Synthea
├── .env                            # Environment variables file (API keys, etc.)
├── .gitignore                      # Git ignore file to prevent certain files from being committed
├── 00. Data Collection & Research.ipynb  # Notebook for gathering clinical trials and research
├── 01. Data Exploration.ipynb             # Notebook exploring and analyzing patient/trial data
├── 02. Setting Up Basic Match.ipynb       # Notebook implementing the initial matching logic
├── Patient Trial Match [Submission].ipynb # Main submission notebook with end-to-end pipeline
├── LLM_PromptExample1.txt                 # Example LLM prompt 1 for clinical trial matching
├── LLM_PromptExample2.txt                 # Example LLM prompt 2 for clinical trial matching
├── requirements.txt                       # Python dependencies for the project
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
