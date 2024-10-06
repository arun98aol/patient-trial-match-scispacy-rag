import os
import pandas as pd
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

def load_nlp_models(model_size='lg'):
    """
    Load and return two NLP models for UMLS linking:
    one for conditions and one for medications, based on the specified model size.

    Parameters:
    model_size (str): Size of the model ('sm', 'md', or 'lg').

    Returns:
    tuple: NLP models for UMLS linking (conditions, medications)
    """
    if model_size not in ['sm', 'md', 'lg']:
        raise ValueError("Invalid model size. Choose from 'sm', 'md', or 'lg'.")
    
    model_name = f"en_core_sci_{model_size}"

    nlp_umls_link = spacy.load(model_name)
    nlp_umls_link.add_pipe("abbreviation_detector")
    nlp_umls_link.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls"
    })

    nlp_rxnorm_link = spacy.load(model_name)
    nlp_rxnorm_link.add_pipe("abbreviation_detector")
    nlp_rxnorm_link.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "rxnorm"
    })

    return nlp_umls_link, nlp_rxnorm_link

def extract_umls_codes(descriptions, nlp):
    """
    Extract UMLS codes from descriptions using the provided NLP model.

    Parameters:
        descriptions (list): List of text descriptions.
        nlp (spacy.lang): NLP model for UMLS linking.

    Returns:
        dict: Mapping from description to list of UMLS codes.
    """
    code_mapping = {}
    for desc in descriptions:
        if pd.isna(desc) or desc == '':
            code_mapping[desc] = []
            continue
        doc = nlp(desc)
        codes = set()
        for entity in doc.ents:
            for kb_ent in entity._.kb_ents:
                concept_id, score = kb_ent
                codes.add(concept_id)
        code_mapping[desc] = list(codes)
    return code_mapping

def extract_umls_code_definitions(unique_codes, nlp_umls_conditions=None, nlp_umls_medications=None, reload=False):
    """
    Extract definitions for UMLS codes using SciSpaCy models.

    Parameters:
        unique_codes (set): Set of unique UMLS codes.
        nlp_umls_conditions (spacy.lang): NLP model for UMLS linking (conditions).
        nlp_umls_medications (spacy.lang): NLP model for UMLS linking (medications).
        reload (bool): Whether to reload the data from the models.

    Returns:
        pd.DataFrame: DataFrame containing code details.
    """

    # check if present in data/umls_codes.csv
    if not reload and os.path.exists('data/umls_codes.csv'):
        umls_codes_df = pd.read_csv('data/umls_codes.csv')
        umls_codes_df = umls_codes_df[umls_codes_df['code'].isin(unique_codes)]
        if len(umls_codes_df) == len(unique_codes):
            return umls_codes_df

    # Initialize a list to store code information
    code_data = []

    # Get the linkers from the models
    linker_conditions = nlp_umls_conditions.get_pipe('scispacy_linker')
    linker_medications = nlp_umls_medications.get_pipe('scispacy_linker')

    # Combine the two linkers' knowledge bases
    all_linkers = [linker_conditions, linker_medications]
    processed_codes = set()

    for code in unique_codes:
        if code in processed_codes:
            continue  # Skip if already processed
        entity = None
        for linker in all_linkers:
            entity = linker.kb.cui_to_entity.get(code)
            if entity:
                break
        if entity:
            code_data.append({
                'code': code,
                'canonical_name': entity.canonical_name,
                'definition': entity.definition,
                'aliases': entity.aliases,
                'types': entity.types
            })
        else:
            # Code not found in either linker
            code_data.append({
                'code': code,
                'canonical_name': None,
                'definition': None,
                'aliases': None,
                'types': None
            })
        processed_codes.add(code)
    
    # Create a DataFrame from the code data
    spacy_umls_codes_df = pd.DataFrame(code_data)

    # Save the DataFrame to a CSV file
    spacy_umls_codes_df.to_csv('data/umls_codes.csv', index=False)
    
    return spacy_umls_codes_df
