import os
import pandas as pd
from .scispacy_models import extract_umls_codes
from datetime import datetime

# Helper functions for data loading and processing
def load_data(file_path, encoding='utf-8'):
    """
    Load data from a CSV file with the specified encoding.

    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    encoding (str, optional): The encoding used to read the CSV file. Default is 'utf-8'.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, encoding=encoding)

# Function to calculate age in years based on birthdate.
def calculate_age(birthdate):
    """
    Calculate age in years based on the provided birthdate.

    Args:
        birthdate (datetime): The birthdate of the individual.

    Returns:
        int: The age in years.
    """
    current_date = datetime.now()
    return (current_date - birthdate).days // 365

def aggregate_list(series):
    """
    Aggregate a pandas Series into a list of unique, non-null values.
    """
    return list(series.dropna().unique())

def convert_to_datetime(df, date_columns):
    """
    Convert specified columns in DataFrame to datetime.
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def aggregate_codes(series):
    """
    Aggregate a pandas Series of lists into a list of unique codes.
    """
    codes = set()
    for items in series.dropna():
        codes.update(items)
    return list(codes)

def create_code_description_map(codes_list, descriptions_list):
    """
    Create a mapping from codes to descriptions.
    """
    code_desc_map = {}
    for codes, desc in zip(codes_list, descriptions_list):
        if isinstance(codes, list):
            for code in codes:
                if code in code_desc_map:
                    code_desc_map[code].add(desc)
                else:
                    code_desc_map[code] = {desc}
    return code_desc_map

# Data Loading and Preprocessing
def prepare_patient_data(nlp_umls_link, nlp_rxnorm_link, reload=False):
    """
    Load and preprocess patient data, conditions, and medications.
    Returns the patient_profiles_grouped DataFrame.

    Parameters:
        nlp_umls_link: NLP model for UMLS linking (conditions).
        nlp_rxnorm_link: NLP model for UMLS linking (medications).
        reload (bool): Whether to reload the data from the original files.
    
    Returns:
        pd.DataFrame: DataFrame containing grouped patient profiles.
    """
    STOP_CODES = {'C3244316', 'C0013227'}
    if not reload and os.path.exists('data/patient/patient_profiles_grouped.csv'):
        return pd.read_csv('data/patient/patient_profiles_grouped.csv')
        
    # Load data
    patients_df = load_data('synthea_100/patients.csv')
    medications_df = load_data('synthea_100/medications.csv')
    conditions_df = load_data('synthea_100/conditions.csv')

    # Calculate age
    patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'], errors='coerce')
    patients_df['AGE'] = patients_df['BIRTHDATE'].apply(calculate_age)

    # Convert date columns to datetime
    date_columns_conditions = ['START', 'STOP']
    date_columns_medications = ['START', 'STOP']
    conditions_df = convert_to_datetime(conditions_df, date_columns_conditions)
    medications_df = convert_to_datetime(medications_df, date_columns_medications)


    # Extract UMLS codes for conditions
    unique_condition_descriptions = conditions_df['DESCRIPTION'].unique()
    condition_description_to_umls_codes = extract_umls_codes(unique_condition_descriptions, nlp_umls_link)
    conditions_df['Condition_UMLS_CODES'] = conditions_df['DESCRIPTION'].map(condition_description_to_umls_codes)

    # Extract UMLS codes for medications
    unique_medication_descriptions = medications_df['DESCRIPTION'].unique()
    medication_description_to_umls_codes = extract_umls_codes(unique_medication_descriptions, nlp_rxnorm_link)
    medications_df['Medication_UMLS_CODES'] = medications_df['DESCRIPTION'].map(medication_description_to_umls_codes)

    conditions_df['Condition_UMLS_CODES'] = conditions_df['Condition_UMLS_CODES'].apply(
        lambda codes: [code for code in codes if code not in STOP_CODES]
    )

    medications_df['Medication_UMLS_CODES'] = medications_df['Medication_UMLS_CODES'].apply(
        lambda codes: [code for code in codes if code not in STOP_CODES]
    )

    # Merge DataFrames
    patients_conditions = pd.merge(patients_df, conditions_df, left_on='Id', right_on='PATIENT', how='left')
    conditions_medications = pd.merge(patients_conditions, medications_df, left_on='ENCOUNTER', right_on='ENCOUNTER', how='left')

    # Create Patient Profiles
    patient_profiles = conditions_medications[['Id', 'AGE', 'GENDER', 
                                               'DESCRIPTION_x', 'CODE_x', 'START_x', 'STOP_x', 'Condition_UMLS_CODES',
                                               'DESCRIPTION_y', 'CODE_y', 'START_y', 'STOP_y', 'Medication_UMLS_CODES',
                                               'ENCOUNTER']]

    # Rename columns
    patient_profiles.columns = [
        'PatientID', 'Age', 'Gender', 
        'Condition_Description', 'Condition_Code', 'Condition_Start', 'Condition_End', 'Condition_UMLS_CODES',
        'Medication_Description', 'Medication_Code', 'Medication_Start', 'Medication_End', 'Medication_UMLS_CODES',
        'Encounter'
    ]

    # Create Patient Profiles Grouped
    patient_data_list = []
    patient_groups = patient_profiles.groupby('PatientID')

    for patient_id, group in patient_groups:
        age = group['Age'].iloc[0]
        gender = group['Gender'].iloc[0]
        condition_descriptions = aggregate_list(group['Condition_Description'])
        condition_codes = aggregate_list(group['Condition_Code'])
        condition_umls_codes = aggregate_codes(group['Condition_UMLS_CODES'])
        medication_descriptions = aggregate_list(group['Medication_Description'])
        medication_codes = aggregate_list(group['Medication_Code'])
        medication_umls_codes = aggregate_codes(group['Medication_UMLS_CODES'])

        # Create code-description maps
        condition_code_desc_map = create_code_description_map(group['Condition_UMLS_CODES'], group['Condition_Description'])
        medication_code_desc_map = create_code_description_map(group['Medication_UMLS_CODES'], group['Medication_Description'])

        patient_data_list.append({
            'PatientID': patient_id,
            'Age': age,
            'Gender': gender,
            'Condition_Description': condition_descriptions,
            'Condition_Code': condition_codes,
            'Condition_UMLS_CODES': condition_umls_codes,
            'Condition_Code_Description_Map': condition_code_desc_map,
            'Medication_Description': medication_descriptions,
            'Medication_Code': medication_codes,
            'Medication_UMLS_CODES': medication_umls_codes,
            'Medication_Code_Description_Map': medication_code_desc_map
        })

    patient_profiles_grouped = pd.DataFrame(patient_data_list)

    # save both dataframes to csv
    patient_profiles_grouped.to_csv('data/patient/patient_profiles_grouped.csv', index=False)
    conditions_medications.to_csv('data/annotated_synthea/conditions_medications.csv', index=False)

    return patient_profiles_grouped