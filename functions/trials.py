import requests
import pandas as pd
import logging
from tqdm import tqdm
import os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
from .scispacy_models import extract_umls_codes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compile regex patterns once at the module level
CRITERIA_SPLIT_PATTERN = re.compile(r'\n(?:\d+\.\s|\*\s|-|\u2022|\u00B7)')
INCLUSION_HEADER = "Inclusion Criteria:"
EXCLUSION_HEADER = "Exclusion Criteria:"

# Define STOP_CODES as a set of UMLS codes to exclude from the final mapping
# Idenitfied from the initial analysis - as commonly misclassified codes
STOP_CODES = {'C3244316', 'C0013227'}

def fetch_n_trials_v2(
        nlp_umls_link,
        nlp_rxnorm_link,
        n=None,
        max_workers=None, 
        verbose=False, 
        reload=True
    ):
    """
    Fetch 'n' clinical trials from ClinicalTrials.gov that are actively recruiting.
    If 'n' is None, fetch all trials that are actively recruiting.
    Extract UMLS codes from inclusion and exclusion criteria using provided NLP models.

    Parameters:
        n (int, optional): Number of trials to fetch. If None, fetch all trials.
        max_workers (int, optional): Maximum number of worker threads to use for parallel processing.
        verbose (bool, optional): If True, set logging level to INFO. If False, set to WARNING.
        reload (bool, optional): If False and processed file exists, load and return it.

    Returns:
        pd.DataFrame: DataFrame containing trial details and extracted UMLS codes.
    """

    # Define the path to save the processed trials
    processed_file_path = os.path.join('data', 'trials', 'df_trials_processed.csv')

    # If reload is False and the processed file exists, load and return it
    if not reload and os.path.exists(processed_file_path):
        return pd.read_csv(processed_file_path)

    # Setup Logging
    logger = logging.getLogger('fetch_n_trials_v2')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    base_url = 'https://clinicaltrials.gov/api/v2/studies'
    page_size = 1000  # Adjusted as per your current function
    query_params = {
        'format': 'json',
        'filter.overallStatus': 'RECRUITING',
        'pageSize': page_size,
        'fields': ','.join([
            'protocolSection.identificationModule.nctId',
            'protocolSection.identificationModule.briefTitle',
            'protocolSection.eligibilityModule.eligibilityCriteria',
            'protocolSection.eligibilityModule.minimumAge',
            'protocolSection.eligibilityModule.maximumAge',
            'protocolSection.eligibilityModule.sex',
            'protocolSection.eligibilityModule.healthyVolunteers'
        ])
    }

    trial_data = []
    total_fetched = 0
    page_token = None
    total_trials_processed = 0
    save_every = 1000  # Save every 1000 trials
    first_save = not os.path.exists(processed_file_path)  # Determine if headers should be written

    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)

        logger.info("Starting to fetch trials...")
        with tqdm(desc="Fetching Trials", unit="page") as fetch_bar:
            while True:
                if page_token:
                    query_params['pageToken'] = page_token

                response = requests.get(base_url, params=query_params)
                response.raise_for_status()
                data = response.json()

                trials = data.get('studies', [])
                if not trials:
                    logger.info("No more trials found.")
                    break

                # Limit the number of trials if 'n' is specified
                if n is not None:
                    remaining = n - total_fetched
                    if remaining <= 0:
                        logger.info("Reached the desired number of trials.")
                        break
                    trials = trials[:remaining]

                total_fetched += len(trials)
                trial_data.extend(trials)

                fetch_bar.update(1)

                # Check if there is a next page
                page_token = data.get('nextPageToken')
                if not page_token:
                    logger.info("No nextPageToken found. All trials fetched.")
                    break

                # Stop fetching if we've reached the desired number of trials
                if n is not None and total_fetched >= n:
                    logger.info("Reached the desired number of trials.")
                    break

        logger.info(f"Total trials fetched: {total_fetched}")

        # Process trials in parallel with progress bar
        def process_trial(trial):
            try:
                nct_id = trial['protocolSection']['identificationModule']['nctId']
                trial_title = trial['protocolSection']['identificationModule']['briefTitle']

                # Extract eligibility criteria
                eligibility_string = trial['protocolSection'].get('eligibilityModule', {}).get('eligibilityCriteria', 'Not Available')
                inclusion_criteria, exclusion_criteria = parse_eligibility_criteria(eligibility_string)

                # Extract additional trial information
                eligibility_module = trial['protocolSection'].get('eligibilityModule', {})
                minimum_age = eligibility_module.get('minimumAge', 'Not Specified')
                maximum_age = eligibility_module.get('maximumAge', 'Not Specified')
                sex = eligibility_module.get('sex', 'Not Specified')
                healthy_volunteers = eligibility_module.get('healthyVolunteers', 'Not Specified')

                # Extract UMLS codes from inclusion and exclusion criteria
                inclusion_umls_codes_mapping = extract_codes_from_criteria(inclusion_criteria, nlp_umls_link, nlp_rxnorm_link)
                exclusion_umls_codes_mapping = extract_codes_from_criteria(exclusion_criteria, nlp_umls_link, nlp_rxnorm_link)

                return {
                    'NCTId': nct_id,
                    'Title': trial_title,
                    'Inclusion_Criteria': inclusion_criteria,
                    'Exclusion_Criteria': exclusion_criteria,
                    'Minimum_Age': minimum_age,
                    'Maximum_Age': maximum_age,
                    'Sex': sex,
                    'Healthy_Volunteers': healthy_volunteers,
                    'Inclusion_Criteria_UMLS_Codes': inclusion_umls_codes_mapping,
                    'Exclusion_Criteria_UMLS_Codes': exclusion_umls_codes_mapping
                }

            except KeyError as ke:
                logger.warning(f"Missing key {ke} in trial data. Skipping trial.")
                return None
            except Exception as e:
                logger.error(f"Error processing trial: {e}")
                return None

        logger.info("Starting parallel processing of trials...")
        save_buffer = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.submit(process_trial, trial) for trial in trial_data)

            # Use tqdm to track progress
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Trials", unit="trial"):
                result = future.result()
                if result:
                    save_buffer.append(result)
                    total_trials_processed += 1

                    # Save every 'save_every' trials
                    if total_trials_processed % save_every == 0:
                        df_save = pd.DataFrame(save_buffer)
                        if first_save:
                            df_save.to_csv(processed_file_path, mode='w', index=False, encoding='utf-8')
                            first_save = False
                            logger.info(f"Saved {save_every} trials to {processed_file_path}.")
                        else:
                            df_save.to_csv(processed_file_path, mode='a', index=False, header=False, encoding='utf-8')
                            logger.info(f"Appended {save_every} trials to {processed_file_path}.")
                        save_buffer = []  # Reset buffer

        logger.info(f"Total trials processed successfully: {total_trials_processed}")

        # Save any remaining trials in the buffer
        if save_buffer:
            df_save = pd.DataFrame(save_buffer)
            if first_save:
                df_save.to_csv(processed_file_path, mode='w', index=False, encoding='utf-8')
                logger.info(f"Saved remaining {len(save_buffer)} trials to {processed_file_path}.")
            else:
                df_save.to_csv(processed_file_path, mode='a', index=False, header=False, encoding='utf-8')
                logger.info(f"Appended remaining {len(save_buffer)} trials to {processed_file_path}.")

        logger.info("DataFrame creation and saving successful.")
        return pd.read_csv(processed_file_path)

    except requests.RequestException as e:
        logger.error(f"Error fetching trials: {e}")
        return None

def parse_eligibility_criteria(eligibility_string: str) -> Tuple[List[str], List[str]]:
    """
    Parse the eligibility string to separate Inclusion and Exclusion criteria.

    Parameters:
        eligibility_string (str): Raw eligibility criteria string.

    Returns:
        tuple: (inclusion_criteria, exclusion_criteria)
    """
    inclusion_criteria = []
    exclusion_criteria = []

    if INCLUSION_HEADER in eligibility_string:
        # Split into inclusion and exclusion sections
        sections = eligibility_string.split(EXCLUSION_HEADER)
        inclusion_section = sections[0].split(INCLUSION_HEADER)[-1]
        exclusion_section = sections[1] if len(sections) > 1 else ""

        # Split by the compiled regex pattern
        inclusion_criteria = CRITERIA_SPLIT_PATTERN.split(inclusion_section)
        exclusion_criteria = CRITERIA_SPLIT_PATTERN.split(exclusion_section)

        # Clean up the criteria using list comprehensions
        inclusion_criteria = [item.strip() for item in inclusion_criteria if item.strip()]
        exclusion_criteria = [item.strip() for item in exclusion_criteria if item.strip()]

    return inclusion_criteria, exclusion_criteria

def extract_codes_from_criteria(criteria_list: List[str],
                                nlp_umls_link,
                                nlp_rxnorm_link) -> Dict[str, List[str]]:
    """
    Extract UMLS codes from a list of criteria using provided NLP models.

    Parameters:
        criteria_list (list): List of criteria strings.
        nlp_umls_link: NLP model for UMLS linking (conditions).
        nlp_rxnorm_link: NLP model for UMLS linking (medications).

    Returns:
        dict: Mapping from criterion string to list of UMLS codes.
    """
    criteria_codes_mapping = {}

    if not criteria_list:
        return criteria_codes_mapping

    # Batch processing for conditions
    condition_codes_dict = extract_umls_codes(criteria_list, nlp_umls_link)
    # Batch processing for medications
    medication_codes_dict = extract_umls_codes(criteria_list, nlp_rxnorm_link)

    for criterion in criteria_list:
        # Extract and deduplicate condition codes
        condition_codes = set(condition_codes_dict.get(criterion, [])) - STOP_CODES

        # Extract and deduplicate medication codes
        medication_codes = set(medication_codes_dict.get(criterion, [])) - STOP_CODES

        # Combine codes
        combined_codes = list(condition_codes.union(medication_codes))
        criteria_codes_mapping[criterion] = combined_codes

    return criteria_codes_mapping
