import json
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from .llm_apis import call_openai


def is_healthy_volunteer(patient):
    """
    Determine if a patient is a healthy volunteer.
    """
    return len(patient['Condition_Description']) == 0

def perform_exclusion_criteria_check(patient, trial):
    """
    Check exclusion criteria and return match status and details.
    """
    exclusion_match = True  # Assume match until proven otherwise
    exclusion_logs = []

    # Prepare patient's condition and medication codes
    patient_condition_codes = set(patient['Condition_UMLS_CODES'])
    patient_medication_codes = set(patient['Medication_UMLS_CODES'])

    # Iterate over exclusion criteria
    for idx, crit in enumerate(trial['Exclusion_Criteria']):
        criterion_match = True  # Assume criterion matches
        criterion_logs = [f"   - Criterion {idx+1}: {crit}"]

        # Get codes associated with the criterion
        criterion_umls_codes = set(trial['Exclusion_Criteria_UMLS_Codes'].get(crit, []))

        # Check condition codes
        overlapping_condition_codes = patient_condition_codes.intersection(criterion_umls_codes)
        if overlapping_condition_codes:
            criterion_match = False
            for code in overlapping_condition_codes:
                descriptions = patient['Condition_Code_Description_Map'].get(code, [])
                descriptions_str = ', '.join(descriptions)
                criterion_logs.append(f"      + [{code}] {descriptions_str}")
        
        # Check medication codes
        overlapping_medication_codes = patient_medication_codes.intersection(criterion_umls_codes)
        if overlapping_medication_codes:
            criterion_match = False
            for code in overlapping_medication_codes:
                descriptions = patient['Medication_Code_Description_Map'].get(code, [])
                descriptions_str = ', '.join(descriptions)
                criterion_logs.append(f"      + [{code}] {descriptions_str}")

        # Append match status for the criterion
        criterion_logs[0] += f" = {'MATCH' if criterion_match else 'NO MATCH'}"
        if not criterion_match:
            exclusion_match = False

        exclusion_logs.extend(criterion_logs)

    return exclusion_match, exclusion_logs

def match_patients_to_trials(
        patient_profiles_grouped,
        df_trials,
        code_definitions_df,
        chunk_size=100,
        debug=False,
        verbose=True,
        max_api_workers=10,
        output_json_format='both',
        patient_start_chunk=0,
    ):
    """
    Match patients to clinical trials based on eligibility criteria, with API integration.

    Parameters:
        patient_profiles_grouped (pd.DataFrame): DataFrame containing patient profiles.
        df_trials (pd.DataFrame): DataFrame containing trial information.
        code_definitions_df (pd.DataFrame): DataFrame with code definitions.
        chunk_size (int): Number of patients and trials to process per chunk.
        debug (bool): If True, generate detailed logs for each patient-trial match.
        verbose (bool): If True, logging and progress bars are enabled.
        max_api_workers (int): Maximum number of concurrent API calls.
        output_json_format (str): Options to include in output ('match', 'no_match', 'both').
        patient_start_chunk (int): Index of the starting patient chunk.

    Returns:
        list: List of dictionaries containing patient IDs and their eligible trials.
    """

    # Set up logging to the specified log file
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"data/output/pt_matches_progress_{datetime_str}.log"

    # Set up the logger for use in Jupyter
    logger = setup_logger_for_jupyter(log_filename, debug=debug)

    if not verbose:
        logger.disabled = True

    # Initialize tqdm with pandas
    tqdm.pandas()

    # Prepare trial data
    df_trials = prepare_trial_data(df_trials)
    logger.info("Trial data prepared.")

    # Initialize the final results list
    final_results = []

    # Mapping to store match results for each patient-trial pair
    match_results_mapping = {}

    patient_chunk_size = 10

    # Process patients in chunks with progress bar
    for patient_start in tqdm(range(patient_start_chunk, len(patient_profiles_grouped), patient_chunk_size), desc="Patient Chunks"):
        patient_chunk = patient_profiles_grouped.iloc[patient_start:patient_start + patient_chunk_size]
        logger.info(f"Processing patient chunk starting at index {patient_start}")

        # Clear final_results to free memory
        final_results.clear()

        # Process trials in chunks with progress bar
        for trial_start in tqdm(range(0, len(df_trials), chunk_size), desc="Trial Chunks", leave=False):
            trial_chunk = df_trials.iloc[trial_start:trial_start + chunk_size]
            logger.info(f"Processing trial chunk starting at index {trial_start}")

            # Create cartesian product and apply matching function
            patient_trial_pairs = patient_chunk.assign(key=1).merge(
                trial_chunk.assign(key=1), on='key').drop('key', axis=1)

            # Apply matching function with progress bar
            match_results = patient_trial_pairs.progress_apply(
                lambda row: evaluate_patient_trial_match_w_api(
                    row, code_definitions_df=code_definitions_df, debug=debug), axis=1)

            patient_trial_pairs['Match_Result'] = match_results

            # Collect results and prepare API call tasks
            for idx, row in patient_trial_pairs.iterrows():
                patient_id = row['PatientID']
                trial_id = row['NCTId']
                match_result = row['Match_Result']

                # Store match result for later use
                match_results_mapping[(patient_id, trial_id)] = match_result
                print('MATCH RESULT:', match_result['is_match'])
                if match_result['is_match']:
                    # Prepare initial eligible trial entry
                    trial_entry = {
                        'trialId': trial_id,
                        'trialName': row['Title'],
                        'eligibilityCriteriaMet': match_result['eligibilityCriteriaMet']
                    }

                    # Add to final results
                    existing_patient = next((item for item in final_results if item['patientId'] == patient_id), None)
                    if existing_patient:
                        existing_patient['eligibleTrials'].append(trial_entry)
                    else:
                        final_results.append({
                            'patientId': patient_id,
                            'eligibleTrials': [trial_entry]
                        })

        # Save results after processing each patient chunk
        output_file_path = f'data/output/patient_trial_matches_chunk_{patient_start}.json'
        save_results_to_json(final_results, output_file_path, logger)

    logger.info("All patient chunks processed. Returning final results.")
    return final_results

def save_results_to_json(results, output_file_path, logger):
    """
    Save the results to a JSON file, appending if the file already exists.

    Parameters:
        results (list): The results to save.
        output_file_path (str): The path to the output JSON file.
        logger (logging.Logger): The logger instance.
    """
    try:
        # Save the results to the JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)

        logger.info(f"Results saved to {output_file_path}.")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file_path}: {e}")

import pandas as pd  # Ensure pandas is installed
import random  # Used in placeholder functions

def update_logs_and_format_output(final_results, api_results, match_results_mapping, output_json_format='both', debug=False):
    """
    Update logs based on API results and format the final JSON output.

    Parameters:
        final_results (list): List of dictionaries containing patient IDs and their eligible trials.
        api_results (dict): Dictionary mapping (patient_id, trial_id) to API output.
        match_results_mapping (dict): Dictionary mapping (patient_id, trial_id) to initial match results.
        output_json_format (str): Options to include in output ('match', 'no_match', 'both').
        debug (bool): If True, additional debug information will be logged.

    Returns:
        list: Updated list of dictionaries containing patient IDs and their eligible trials.
    """
    updated_final_results = []

    # Convert final_results to a dict for faster lookup
    final_results_dict = {entry['patientId']: entry for entry in final_results}

    for (patient_id, trial_id), api_result in api_results.items():
        patient_entry = final_results_dict.get(patient_id, {'patientId': patient_id, 'eligibleTrials': []})
        trial_entry = next((t for t in patient_entry['eligibleTrials'] if t['trialId'] == trial_id), None)

        if trial_entry:
            # Update existing trial
            trial_entry['eligibilityCriteriaMet'] = (
                api_result.get('inclusionCriteriaMet', []) +
                api_result.get('exclusionCriteriaMet', [])
            )
            trial_entry['patientIsEligibleForTrial'] = api_result.get('patientIsEligibleForTrial', False)
            
            if debug:
                print(f"Updated existing trial {trial_id} for patient {patient_id} with eligibility {trial_entry['patientIsEligibleForTrial']}")
        else:
            # Add new trial that wasn't initially matched
            if api_result.get('patientIsEligibleForTrial', False) or output_json_format == 'both':
                new_trial_entry = {
                    'trialId': trial_id,
                    'trialName': api_result.get('trialName', 'Unknown Trial Name'),  # Assuming API provides trialName
                    'eligibilityCriteriaMet': (
                        api_result.get('inclusionCriteriaMet', []) +
                        api_result.get('exclusionCriteriaMet', [])
                    ),
                    'patientIsEligibleForTrial': api_result.get('patientIsEligibleForTrial', False)
                }
                patient_entry['eligibleTrials'].append(new_trial_entry)
                final_results_dict[patient_id] = patient_entry
                
                if debug:
                    print(f"Added new trial {trial_id} for patient {patient_id} with eligibility {new_trial_entry['patientIsEligibleForTrial']}")

        # Update log files based on API results
        # Define the log directory and pattern
        log_dir = os.path.join('logs', patient_id)
        log_files_pattern = os.path.join(log_dir, f"*{trial_id}_*.log")
        log_files = glob(log_files_pattern)
        if log_files:
            original_log_filepath = log_files[0]
            # Read the original log content
            with open(original_log_filepath, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # Append the API response to the logs
            api_check_lines = ["API Check:"]
            # Pretty-print the API response
            for key_resp, value_resp in api_result.items():
                if isinstance(value_resp, list):
                    api_check_lines.append(f"{key_resp}:")
                    for item in value_resp:
                        api_check_lines.append(f"\t- {item}")
                else:
                    api_check_lines.append(f"{key_resp}: {value_resp}")
            api_check_text = '\n'.join(api_check_lines)
            log_content += '\n' + api_check_text

            # Determine if file name needs to be updated
            initial_match_result = match_results_mapping.get((patient_id, trial_id))
            initial_status = 'MATCH' if initial_match_result and initial_match_result.get('is_match') else 'NO_MATCH'
            updated_status = 'MATCH' if api_result.get('patientIsEligibleForTrial', False) else 'NO_MATCH'

            if initial_status != updated_status:
                # Remove the initial status prefix from the filename
                original_log_filename = os.path.basename(original_log_filepath)
                original_log_filename = re.sub(r'^\[.*?\]_', '', original_log_filename)
                # Construct new file name with updated status
                new_log_filename = f'[{updated_status}]_{original_log_filename}'
                new_log_filepath = os.path.join(log_dir, new_log_filename)
                # Rename the file
                os.rename(original_log_filepath, new_log_filepath)
                # Write the updated log content to the renamed file
                with open(new_log_filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
            else:
                # Status didn't change, overwrite the original file
                with open(original_log_filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
        else:
            # Log file not found; handle this case if necessary
            if debug:
                print(f"Log file not found for patient {patient_id} and trial {trial_id}.")

    # Convert back to list
    updated_final_results = list(final_results_dict.values())
    
    return updated_final_results

def setup_logger_for_jupyter(log_filename, debug=False):
    """
    Sets up the logger to log into a file in a Jupyter environment.
    Removes existing handlers to avoid conflicts.
    """
    # Create a logger
    logger = logging.getLogger('match_patients_to_trials_w_api')

    # Clear existing handlers to prevent multiple outputs in Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set logging level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create a file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger (logging only to the file now)
    logger.addHandler(fh)

    return logger

def match_patients_to_trials_w_api(
        patient_profiles_grouped,
        df_trials,
        code_definitions_df,
        chunk_size=100,
        debug=False,
        verbose=True,
        max_api_workers=10,
        output_json_format='both'
    ):
    """
    Match patients to clinical trials based on eligibility criteria, with API integration.

    Parameters:
        patient_profiles_grouped (pd.DataFrame): DataFrame containing patient profiles.
        df_trials (pd.DataFrame): DataFrame containing trial information.
        code_definitions_df (pd.DataFrame): DataFrame with code definitions.
        chunk_size (int): Number of patients and trials to process per chunk.
        debug (bool): If True, generate detailed logs for each patient-trial match.
        verbose (bool): If True, logging and progress bars are enabled.
        max_api_workers (int): Maximum number of concurrent API calls.
        output_json_format (str): Options to include in output ('match', 'no_match', 'both').

    Returns:
        list: List of dictionaries containing patient IDs and their eligible trials.
    """

    # Set up logging to the specified log file
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"data/output/pt_matches_progress_{datetime_str}.log"

    # Set up the logger for use in Jupyter
    logger = setup_logger_for_jupyter(log_filename, debug=debug)

    if not verbose:
        logger.disabled = True

    # Initialize tqdm with pandas
    tqdm.pandas()

    # Prepare trial data
    df_trials = prepare_trial_data(df_trials)
    logger.info("Trial data prepared.")

    # Initialize the final results list
    final_results = []

    # Initialize list to collect API call tasks
    api_call_tasks = []

    # Mapping to store match results for each patient-trial pair
    match_results_mapping = {}

    patient_chunk_size = 50

    # Process patients in chunks with progress bar
    for patient_start in tqdm(range(0, len(patient_profiles_grouped), patient_chunk_size), desc="Patient Chunks"):
        patient_chunk = patient_profiles_grouped.iloc[patient_start:patient_start + patient_chunk_size]
        logger.info(f"Processing patient chunk starting at index {patient_start}")

        # Process trials in chunks with progress bar
        for trial_start in tqdm(range(0, len(df_trials), chunk_size), desc="Trial Chunks", leave=False):
            trial_chunk = df_trials.iloc[trial_start:trial_start + chunk_size]
            logger.info(f"Processing trial chunk starting at index {trial_start}")

            # Create cartesian product and apply matching function
            patient_trial_pairs = patient_chunk.assign(key=1).merge(
                trial_chunk.assign(key=1), on='key').drop('key', axis=1)

            # Apply matching function with progress bar
            match_results = patient_trial_pairs.progress_apply(
                lambda row: evaluate_patient_trial_match_w_api(
                    row, code_definitions_df=code_definitions_df, debug=debug), axis=1)

            patient_trial_pairs['Match_Result'] = match_results

            # Collect results and prepare API call tasks
            for idx, row in patient_trial_pairs.iterrows():
                patient_id = row['PatientID']
                trial_id = row['NCTId']
                match_result = row['Match_Result']

                # Store match result for later use
                match_results_mapping[(patient_id, trial_id)] = match_result

                if match_result['is_match']:
                    # Prepare initial eligible trial entry
                    trial_entry = {
                        'trialId': trial_id,
                        'trialName': row['Title'],
                        'eligibilityCriteriaMet': match_result['eligibilityCriteriaMet'],
                        'patientIsEligibleForTrial': True  # Initial match is True
                    }

                    # Add to final results
                    existing_patient = next((item for item in final_results if item['patientId'] == patient_id), None)
                    if existing_patient:
                        existing_patient['eligibleTrials'].append(trial_entry)
                    else:
                        final_results.append({
                            'patientId': patient_id,
                            'eligibleTrials': [trial_entry]
                        })

                    if debug:
                        logger.debug(f"Initially matched trial {trial_id} for patient {patient_id}")

                # Check if API call is needed
                if match_result['need_api_call']:
                    # Prepare arguments for the API call
                    log_text = match_result['log_text']
                    # Enqueue API call
                    api_call_tasks.append((patient_id, trial_id, log_text))

    output_file_path = 'data/output/patient_trial_matches.json'
    save_results_to_json(final_results, output_file_path, logger)

    # Log the number of API calls to be made
    total_api_calls = len(api_call_tasks)
    logger.info(f"Total API calls to be made: {total_api_calls}")

    # Process API calls in parallel with progress bar
    api_results = {}  # To store API results
    api_call_counter = 0  # To count the number of API calls made

    if api_call_tasks:
        def process_api_call(task):
            patient_id, trial_id, log_text = task
            # Call the OpenAI API
            api_result = call_openai(log_text)  # Replace with your actual API call
            # Return the result along with identifiers
            return patient_id, trial_id, api_result

        with ThreadPoolExecutor(max_workers=max_api_workers) as executor:
            futures = [executor.submit(process_api_call, task) for task in api_call_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="API Calls"):
                try:
                    patient_id, trial_id, api_result = future.result()
                    logger.info(f"API call completed for patient {patient_id} and trial {trial_id} with result {api_result.get('patientIsEligibleForTrial')}")
                    
                    # Store the API result
                    api_results[(patient_id, trial_id)] = api_result

                    # Increment API call counter
                    api_call_counter += 1

                    # Every 1000 API calls, save the current results
                    if api_call_counter % 1000 == 0 or api_call_counter == total_api_calls:
                        logger.info(f"Saving results after {api_call_counter} API calls.")
                        # Update logs and format output based on API results
                        temp_results = update_logs_and_format_output(
                            final_results=final_results,
                            api_results=api_results,
                            match_results_mapping=match_results_mapping,
                            output_json_format=output_json_format,
                            debug=debug
                        )
                        # Save the updated results to the JSON file
                        save_results_to_json(temp_results, output_file_path, logger)
                        # Update final_results with temp_results
                        final_results = temp_results
                        # Clear api_results to prevent re-processing
                        api_results.clear()

                except Exception as e:
                    logger.error(f"API call failed: {e}")

        # After all API calls are done, ensure all results are saved
        logger.info("All API calls completed. Saving final results.")
        final_results = update_logs_and_format_output(
            final_results=final_results,
            api_results=api_results,
            match_results_mapping=match_results_mapping,
            output_json_format=output_json_format,
            debug=debug
        )
        save_results_to_json(final_results, output_file_path, logger)

    else:
        logger.info("No API calls needed.")
        # Save the final results without API updates
        save_results_to_json(final_results, output_file_path, logger)

    return final_results

def evaluate_patient_trial_match_w_api(row, code_definitions_df, debug=False):
    """
    Evaluate if a patient matches a trial's eligibility criteria with detailed messages.

    Parameters:
        row (pd.Series): Series containing patient and trial data.
        code_definitions_df (pd.DataFrame): DataFrame with code definitions.
        debug (bool): If True, generate logs for the match evaluation.

    Returns:
        dict: Dictionary containing match status, detailed criteria messages, and log text.
    """
    import os
    from datetime import datetime
    import numpy as np

    # print('ROw:', row)

    # Extract patient and trial data from the row
    patient_id = row['PatientID']
    patient_age = row['Age']
    patient_gender = row['Gender']
    is_healthy = is_healthy_volunteer_row(row)
    trial_id = row['NCTId']
    trial_title = row['Title']
    min_age = row['Minimum_Age']
    max_age = row['Maximum_Age']
    trial_sex = row['Sex']
    healthy_volunteers = row['Healthy_Volunteers']
    inclusion_criteria = row['Inclusion_Criteria']
    exclusion_criteria = row['Exclusion_Criteria']
    inclusion_codes_mapping = row['Inclusion_Criteria_UMLS_Codes']
    exclusion_codes_mapping = row['Exclusion_Criteria_UMLS_Codes']
    condition_codes = row['Condition_UMLS_CODES']
    medication_codes = row['Medication_UMLS_CODES']
    condition_desc_map = row['Condition_Code_Description_Map']
    medication_desc_map = row['Medication_Code_Description_Map']

    # Initialize match status and eligibility criteria messages
    is_match = True
    eligibility_criteria_met = []
    eligibility_criteria_not_met = []

    # Initialize logs
    log_lines = []

    # 1. Age check
    # print('MIN AGE:', min_age, ', MAX AGE:', max_age)
    min_age_display = min_age if not np.isnan(min_age) else 'Not Specified'
    max_age_display = max_age if not np.isnan(max_age) else 'Not Specified'
    age_match = is_age_match(patient_age, min_age, max_age)
    if age_match:
        message = f"Patient Age ({patient_age}) matches Expected Range ({min_age_display}, {max_age_display})"
        eligibility_criteria_met.append(message)
    else:
        message = f"Patient Age ({patient_age}) does not match Expected Range ({min_age_display}, {max_age_display})"
        eligibility_criteria_not_met.append(message)
        is_match = False
    log_lines.append(f"1. {message}")

    # 2. Sex check
    trial_sex_display = trial_sex if trial_sex else 'Not Specified'
    sex_match = is_sex_match(patient_gender, trial_sex)
    if sex_match:
        message = f"Patient Sex ({patient_gender}) matches Trial Sex Requirement ({trial_sex_display})"
        eligibility_criteria_met.append(message)
    else:
        message = f"Patient Sex ({patient_gender}) does not match Trial Sex Requirement ({trial_sex_display})"
        eligibility_criteria_not_met.append(message)
        is_match = False
    log_lines.append(f"2. {message}")

    # 3. Health status check
    health_status_display = 'Healthy Volunteers' if healthy_volunteers else 'Patients'
    health_match = True
    if healthy_volunteers and not is_healthy:
        health_match = False
    if health_match:
        message = f"Patient Health Status ({'Healthy' if is_healthy else 'Not Healthy'}) matches Trial Requirement ({health_status_display})"
        eligibility_criteria_met.append(message)
    else:
        message = f"Patient Health Status ({'Healthy' if is_healthy else 'Not Healthy'}) does not match Trial Requirement ({health_status_display})"
        eligibility_criteria_not_met.append(message)
        is_match = False
    log_lines.append(f"3. {message}")

    # Initialize variable to indicate if we need to call API
    need_api_call = False

    # Proceed to Exclusion Criteria check
    exclusion_match = True
    exclusion_logs = []
    num_exclusion_reasons = 0
    overlapping_codes = set()

    # Patient codes as sets
    patient_condition_codes = set(condition_codes)
    patient_medication_codes = set(medication_codes)

    # Iterate over exclusion criteria
    for idx, crit in enumerate(exclusion_criteria):
        criterion_match = True  # Assume criterion matches
        criterion_logs = [f"   - Criterion {idx+1}: {crit}"]

        # Get codes associated with the criterion
        criterion_codes = set(exclusion_codes_mapping.get(crit, []))

        # Check condition codes
        overlapping_condition_codes = patient_condition_codes.intersection(criterion_codes)
        if overlapping_condition_codes:
            criterion_match = False
            num_exclusion_reasons += 1
            overlapping_codes.update(overlapping_condition_codes)
            for code in overlapping_condition_codes:
                descriptions = condition_desc_map.get(code, [])
                descriptions_str = ', '.join(descriptions) if descriptions else ''
                criterion_logs.append(f"      + Overlapping Condition Code [{code}]: {descriptions_str}")

        # Check medication codes
        overlapping_medication_codes = patient_medication_codes.intersection(criterion_codes)
        if overlapping_medication_codes:
            criterion_match = False
            num_exclusion_reasons += 1
            overlapping_codes.update(overlapping_medication_codes)
            for code in overlapping_medication_codes:
                descriptions = medication_desc_map.get(code, [])
                descriptions_str = ', '.join(descriptions) if descriptions else ''
                criterion_logs.append(f"      + Overlapping Medication Code [{code}]: {descriptions_str}")

        # Append match status for the criterion
        if not criterion_match:
            message = f"Patient has overlapping conditions/medications with exclusion criterion: {crit}"
            eligibility_criteria_not_met.append(message)
            exclusion_match = False
        else:
            message = f"Patient has no overlapping conditions/medications with exclusion criterion: {crit}"
            eligibility_criteria_met.append(message)

        criterion_logs[0] += f" = {'MATCH' if criterion_match else 'NO MATCH'}"
        exclusion_logs.extend(criterion_logs)

    log_lines.append("4. Exclusion Criteria:")
    log_lines.extend(exclusion_logs)

    if exclusion_match:
        log_lines.append("   - Exclusion Criteria Check: MATCH")
        # Since there is a match, we need to call the API to fact-check
        need_api_call = True
    else:
        log_lines.append("   - Exclusion Criteria Check: NO MATCH")
        if num_exclusion_reasons < 2:
            # Less than 2 reasons for exclusion, call the API to fact-check
            need_api_call = True
        else:
            need_api_call = False
        is_match = False  # Patient is not eligible due to exclusion criteria

    # Combine log lines into 'log_text'
    log_text = '\n'.join(log_lines)

    # If debug is True, write logs to file
    if debug:
        # Prepare patient-specific log directory
        patient_log_dir = os.path.join('logs', patient_id)
        os.makedirs(patient_log_dir, exist_ok=True)

        # Prepare log file name
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{trial_id}_{datetime_str}.log"
        if is_match:
            log_filename = f"[MATCH]_{log_filename}"

        # Write logs to file
        log_filepath = os.path.join(patient_log_dir, log_filename)
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write(log_text)

    # Return the updated match result dictionary
    return {
        'is_match': is_match,
        'eligibilityCriteriaMet': eligibility_criteria_met,
        'eligibilityCriteriaNotMet': eligibility_criteria_not_met,
        'need_api_call': need_api_call,
        'log_text': log_text
    }

def perform_exclusion_criteria_check_row_w_api(row):
    """
    Check exclusion criteria for a patient-trial pair.

    Parameters:
        row (pd.Series): Series containing patient and trial data.

    Returns:
        tuple: (exclusion_match (bool), exclusion_logs (list), num_exclusion_reasons (int), overlapping_codes (set))
    """
    # Initialize match status
    exclusion_match = True  # Assume match until proven otherwise
    exclusion_logs = []
    num_exclusion_reasons = 0
    overlapping_codes = set()  # To collect overlapping UMLS codes

    # Extract necessary data from row
    exclusion_criteria = row['Exclusion_Criteria']
    exclusion_codes_mapping = row['Exclusion_Criteria_UMLS_Codes']
    condition_codes = set(row['Condition_UMLS_CODES'])
    medication_codes = set(row['Medication_UMLS_CODES'])
    condition_desc_map = row['Condition_Code_Description_Map']
    medication_desc_map = row['Medication_Code_Description_Map']

    # Iterate over exclusion criteria
    for idx, crit in enumerate(exclusion_criteria):
        criterion_match = True  # Assume criterion matches
        criterion_logs = [f"   - Criterion {idx+1}: {crit}"]

        # Get codes associated with the criterion
        criterion_codes = set(exclusion_codes_mapping.get(crit, []))

        # Check for overlapping condition codes
        overlapping_condition_codes = condition_codes.intersection(criterion_codes)
        if overlapping_condition_codes:
            criterion_match = False
            num_exclusion_reasons += 1
            overlapping_codes.update(overlapping_condition_codes)
            for code in overlapping_condition_codes:
                descriptions = condition_desc_map.get(code, [])
                descriptions_str = ', '.join(descriptions) if descriptions else ''
                criterion_logs.append(f"      + [{code}] {descriptions_str}")

        # Check for overlapping medication codes
        overlapping_medication_codes = medication_codes.intersection(criterion_codes)
        if overlapping_medication_codes:
            criterion_match = False
            num_exclusion_reasons += 1
            overlapping_codes.update(overlapping_medication_codes)
            for code in overlapping_medication_codes:
                descriptions = medication_desc_map.get(code, [])
                descriptions_str = ', '.join(descriptions) if descriptions else ''
                criterion_logs.append(f"      + [{code}] {descriptions_str}")

        # Append match status for the criterion
        criterion_logs[0] += f" = {'MATCH' if criterion_match else 'NO MATCH'}"
        if not criterion_match:
            exclusion_match = False

        exclusion_logs.extend(criterion_logs)

    return exclusion_match, exclusion_logs, num_exclusion_reasons, overlapping_codes

def evaluate_patient_trial_match(row, debug=False):
    """
    Evaluate if a patient matches a trial's eligibility criteria.

    Parameters:
        row (pd.Series): Series containing patient and trial data.
        debug (bool): If True, generate logs for the match evaluation.

    Returns:
        dict: Dictionary containing match status and criteria met.
    """
    # Extract patient and trial data from the row
    patient_id = row['PatientID']
    patient_age = row['Age']
    patient_gender = row['Gender']
    is_healthy = is_healthy_volunteer_row(row)
    trial_id = row['NCTId']
    trial_title = row['Title']
    min_age = row['Minimum_Age']
    max_age = row['Maximum_Age']
    trial_sex = row['Sex']
    healthy_volunteers = row['Healthy_Volunteers']

    # Initialize match status and eligibility criteria met
    is_match = True
    eligibility_criteria_met = []

    # Initialize logs if debug is True
    log_lines = []
    if debug:
        log_lines.append(f"Patient ID: {patient_id}")
        log_lines.append(f"Trial ID: {trial_id}")
        log_lines.append("Match Check:")

    # 1. Age check
    age_match = is_age_match(patient_age, min_age, max_age)
    if age_match:
        eligibility_criteria_met.append('Age')
    else:
        is_match = False
    if debug:
        log_lines.append(f"1. Age = {patient_age} vs [{min_age}, {max_age}] = {'MATCH' if age_match else 'NO MATCH'}")

    # 2. Sex check
    sex_match = is_sex_match(patient_gender, trial_sex)
    if sex_match:
        eligibility_criteria_met.append('Sex')
    else:
        is_match = False
    if debug:
        log_lines.append(f"2. Sex = {patient_gender} vs {trial_sex} = {'MATCH' if sex_match else 'NO MATCH'}")

    # 3. Health check
    health_match = True
    if healthy_volunteers and not is_healthy:
        health_match = False
    if health_match:
        eligibility_criteria_met.append('Health Status')
    else:
        is_match = False
    if debug:
        log_lines.append(f"3. Health = {'Healthy' if is_healthy else 'Not Healthy'} vs {'Healthy Volunteers' if healthy_volunteers else 'Patients'} = {'MATCH' if health_match else 'NO MATCH'}")

    # Proceed only if basic checks passed
    if is_match:
        # 4. Exclusion Criteria check
        exclusion_match, exclusion_logs = perform_exclusion_criteria_check_row(row)
        if exclusion_match:
            eligibility_criteria_met.append('Exclusion Criteria')
        else:
            is_match = False
        if debug:
            log_lines.append("4. Exclusion Criteria:")
            log_lines.extend(exclusion_logs)

    # Generate logs if debug is True
    if debug:
        # Prepare patient-specific log directory
        patient_log_dir = os.path.join('logs', patient_id)
        os.makedirs(patient_log_dir, exist_ok=True)

        # Prepare log file name
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{trial_id}_{datetime_str}.log"
        if is_match:
            log_filename = f"[MATCH]_{log_filename}"

        # Write logs to file
        log_filepath = os.path.join(patient_log_dir, log_filename)
        with open(log_filepath, 'w', encoding='utf-8') as f:
            for line in log_lines:
                f.write(line + '\n')

    return {'is_match': is_match, 'eligibilityCriteriaMet': eligibility_criteria_met}

def is_healthy_volunteer_row(row):
    """
    Determine if a patient is a healthy volunteer based on row data.
    """
    return len(row['Condition_Description']) == 0

def perform_exclusion_criteria_check_row(row):
    """
    Check exclusion criteria for a patient-trial pair.

    Parameters:
        row (pd.Series): Series containing patient and trial data.

    Returns:
        tuple: (exclusion_match (bool), exclusion_logs (list))
    """
    exclusion_match = True  # Assume match until proven otherwise
    exclusion_logs = []

    # Patient codes
    patient_condition_codes = set(row['Condition_UMLS_CODES'])
    patient_medication_codes = set(row['Medication_UMLS_CODES'])

    # Iterate over exclusion criteria
    exclusion_criteria = row['Exclusion_Criteria']
    exclusion_codes_mapping = row['Exclusion_Criteria_UMLS_Codes']

    for idx, crit in enumerate(exclusion_criteria):
        criterion_match = True  # Assume criterion matches
        criterion_logs = [f"   - Criterion {idx+1}: {crit}"]

        # Get codes associated with the criterion
        criterion_codes = set(exclusion_codes_mapping.get(crit, []))

        # Check condition codes
        overlapping_condition_codes = patient_condition_codes.intersection(criterion_codes)
        if overlapping_condition_codes:
            criterion_match = False
            for code in overlapping_condition_codes:
                descriptions = row['Condition_Code_Description_Map'].get(code, [])
                descriptions_str = ', '.join(descriptions)
                criterion_logs.append(f"      + [{code}] {descriptions_str}")

        # Check medication codes
        overlapping_medication_codes = patient_medication_codes.intersection(criterion_codes)
        if overlapping_medication_codes:
            criterion_match = False
            for code in overlapping_medication_codes:
                descriptions = row['Medication_Code_Description_Map'].get(code, [])
                descriptions_str = ', '.join(descriptions)
                criterion_logs.append(f"      + [{code}] {descriptions_str}")

        # Append match status for the criterion
        criterion_logs[0] += f" = {'MATCH' if criterion_match else 'NO MATCH'}"
        if not criterion_match:
            exclusion_match = False

        exclusion_logs.extend(criterion_logs)

    return exclusion_match, exclusion_logs

def prepare_trial_data(df_trials):
    """
    Preprocess trial data to ensure proper data types.
    """
    # Convert age fields to integers
    df_trials['Minimum_Age'] = df_trials['Minimum_Age'].apply(convert_age)
    df_trials['Maximum_Age'] = df_trials['Maximum_Age'].apply(convert_age)
    # Ensure 'Healthy_Volunteers' is boolean
    df_trials['Healthy_Volunteers'] = df_trials['Healthy_Volunteers'].apply(lambda x: x.lower() == 'yes' if isinstance(x, str) else False)
    return df_trials

def convert_age(age_str):
    """
    Convert age strings to integers.
    """
    # if age is not str
    if not isinstance(age_str, str):
        return age_str
    if pd.isna(age_str) or age_str in ['Not Specified', 'N/A']:
        return np.nan
    try:
        return int(age_str.split()[0])
    except:
        return np.nan

def is_age_match(patient_age, min_age, max_age):
    """
    Check if patient's age matches trial's age criteria.
    """
    if pd.isna(min_age):
        min_age = 0
    if pd.isna(max_age):
        max_age = 120
    return min_age <= patient_age <= max_age

def is_sex_match(patient_gender, trial_gender):
    """
    Check if patient's gender matches trial's sex criteria.
    """
    if trial_gender.upper() == "ALL":
        return True
    return patient_gender.upper() == trial_gender.upper()
