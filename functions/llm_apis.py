import functools
import json
import os
from typing import Dict, List

from mistralai import Mistral
from mistralai.models.function import Function
from mistralai.models.systemmessage import SystemMessage
from mistralai.models.usermessage import UserMessage
from mistralai.models.toolmessage import ToolMessage

from openai import OpenAI, AzureOpenAI
import openai
from retry import retry   

# dotenv setup to load environment variables
from dotenv import load_dotenv
load_dotenv()


def call_mistral_api(log_text: str) -> Dict:
    """
    Call the Mistral AI API with the given log text and retrieve eligibility information.

    Parameters:
        log_text (str): The log text generated during patient-trial matching.

    Returns:
        Dict: Parsed JSON response containing 'patientIsEligibleForTrial' (bool),
              'inclusionCriteriaMet' (list), and 'exclusionCriteriaMet' (list).
    """
    # Define the function that the model can call
    tools = [
        {
            "type": "function",
            "function": Function(
                name="evaluate_patient_eligibility",
                description="Determine if the patient is eligible for the trial based on the log provided.",
                parameters={
                    "type": "object",
                    "properties": {
                        "inclusionCriteriaMet": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "Description of the inclusion criterion and why/why not it was met. This explanation should be concise and accurate."
                            },
                            "description": "List of inclusion criteria met by the patient."
                        },
                        "exclusionCriteriaMet": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "Description of the exclusion criterion and why/why not it was met. This explanation should be concise and accurate."
                            },
                            "description": "List of exclusion criteria met by the patient."
                        },
                        "patientIsEligibleForTrial": {
                            "type": "boolean",
                            "description": "Indicates if the patient is eligible for the trial. This should be an accurate assessment based on the provided log."
                        }
                    },
                    "required": ["inclusionCriteriaMet", "exclusionCriteriaMet", "patientIsEligibleForTrial"]
                }
            )
        }
    ]

    # Initialize Mistral client
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")

    model = "mistral-small-latest"  # Using the cheapest available model

    client = Mistral(api_key=api_key)

    # Prepare initial messages
    messages = [SystemMessage(content="""Analyze the following patient-trial matching log and determine eligibility. 
    **Patient-Trial Matching Log Format:**
    Patient ID: string
    Trial ID: string
    Match Check:
    1. Age = {{patient_age}} vs {{trial_age_required}} = MATCH (trial age can be nan if not specified)
    2. Sex = {{patient_sex}} vs {{trial_sex_required}} = MATCH
    3. Health = {{patient_health_status}} vs {{trial_health_status_required}} = MATCH
    4. Exclusion Criteria:
    - Criterion 1: {{trial_exclusion_criterion_1}} = MATCH (match if patient doesn't have the condition, i.e., MATCH implies the patient is still eligible)
        ...
    - Criterion n-1: {{trial_exclusion_criterion_n-1}} = NO MATCH (no match if patient has the condition, i.e., NO MATCH implies the patient is not eligible)
        + [UMLS_CODE_1] Condition 1 (finding) (UMLS_CODE_1 is the UMLS code for the condition) (these codes and the condition describe why the patient is not eligible)
        + [UMLS_CODE_2] Condition 2 (finding) ...
    - Criterion n: {{trial_exclusion_criterion_n}} = MATCH 

        The log contains a simple analysis conducted using SciSpaCy models and their NER capabilities. However, these results may be inconsistent due to false positives. As a Healthcare QC Specialist, you need to review the log and provide a final assessment of the patient's eligibility for the clinical trial.

        **Example Output JSON Format:**
        {
            "inclusionCriteriaMet": ["Patient meets the age criteria.", ...],
            "exclusionCriteriaMet": ["Patient does not have Summarize Condition 1 (UMLS_CODE_1) as required.", ...],
            "patientIsEligibleForTrial": true/false
        }
    """)]

    # Append the log text as user message
    messages.append(UserMessage(content=log_text))

    # Send the request to Mistral API with tools (which define the output format)
    response = client.chat.complete(model=model, messages=messages, tools=tools)

    # Handle response and ensure it returns a JSON format with patient eligibility
    try:
        if response.choices and response.choices[0].message.tool_calls:
            # Check if the model invoked the tool and produced the required format
            tool_call = response.choices[0].message.tool_calls[0]
            function_params = json.loads(tool_call.function.arguments)
            return function_params
        else:
            # Try to parse the response directly as JSON if no tool call
            eligibility_json = json.loads(response.choices[0].message.content)
            return eligibility_json
    except json.JSONDecodeError:
        print("Failed to parse JSON output from Mistral API.")
        return {}
        
def call_openai(log_text: str) -> Dict:
    """
    Call the OpenAI API with the given log text and retrieve eligibility information.

    Parameters:
        log_text (str): The log text generated during patient-trial matching.

    Returns:
        Dict: Parsed JSON response containing 'patientIsEligibleForTrial' (bool),
              'inclusionCriteriaMet' (list), and 'exclusionCriteriaMet' (list).
    """


    # Define the function that the model can call
    functions = [
        {
        "type": "function",
        "function":
            {
            "name": "evaluate_patient_eligibility",
            "description": "Determine if the patient is eligible for the trial based on the log provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inclusionCriteriaMet": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Description of the inclusion criterion and why/why not it was met."
                        },
                        "description": "List of inclusion criteria met by the patient."
                    },
                    "exclusionCriteriaMet": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Description of the exclusion criterion and why/why not it was met."
                        },
                        "description": "List of exclusion criteria met by the patient."
                    },
                    "patientIsEligibleForTrial": {
                        "type": "boolean",
                        "description": "Indicates if the patient is eligible for the trial."
                    }
                },
                "required": ["inclusionCriteriaMet", "exclusionCriteriaMet", "patientIsEligibleForTrial"]
            }
        }
        }
    ]

    # Prepare the messages for the conversation
    messages = [
        {"role": "system", "content": """
        Analyze the following patient-trial matching log and determine eligibility. 
    **Patient-Trial Matching Log Format:**
    Patient ID: string
    Trial ID: string
    Match Check:
    1. Age = {{patient_age}} vs {{trial_age_required}} = MATCH (trial age can be nan if not specified)
    2. Sex = {{patient_sex}} vs {{trial_sex_required}} = MATCH
    3. Health = {{patient_health_status}} vs {{trial_health_status_required}} = MATCH
    4. Exclusion Criteria:
    - Criterion 1: {{trial_exclusion_criterion_1}} = MATCH (match if patient doesn't have the condition, i.e., MATCH implies the patient is still eligible)
        ...
    - Criterion n-1: {{trial_exclusion_criterion_n-1}} = NO MATCH (no match if patient has the condition, i.e., NO MATCH implies the patient is not eligible)
        + [UMLS_CODE_1] Condition 1 (finding) (UMLS_CODE_1 is the UMLS code for the condition) (these codes and the condition describe why the patient is not eligible)
        + [UMLS_CODE_2] Condition 2 (finding) ...
    - Criterion n: {{trial_exclusion_criterion_n}} = MATCH 

        The log contains a simple analysis conducted using SciSpaCy models and their NER capabilities. However, these results may be inconsistent due to false positives. As a Healthcare QC Specialist, you need to review the log and provide a final assessment of the patient's eligibility for the clinical trial.

        **Example Output JSON Format:**
        {
            "inclusionCriteriaMet": ["Patient meets the age criteria.", ...],
            "exclusionCriteriaMet": ["Patient does not have Summarize Condition 1 (UMLS_CODE_1) as required.", ...],
            "patientIsEligibleForTrial": true/false
        }
        Note: The "+ [UMLS_CODE] Condition" lines could be flawed due to false positives, that's when you need to correct the log. One way to find these false positives is to check if description of criteria n matches the conditions described by the codes. If not, it is a false positive.
        Finally, make modifications to the log (if necessary) to ensure the patient's eligibility is accurately assessed. Watch out for false positives and negatives in the criteria analysis. Use chain-of-thought reasoning to spot double negatives and other complex patterns.
        """},
        {"role": "user", "content": log_text}
    ]

    # Call the OpenAI API
    response_json = call_openai_api(
        model="cx-gpt-4o",
        seed=274,
        bw_message=messages,
        tools=functions,
        tool_choice= {"type": "function", "function": {"name": "evaluate_patient_eligibility"}},
        temperature=1,
        max_tokens=500
    )

    return response_json        
import time

@retry(tries=3, delay=2)  
def call_openai_api(  
        model: str,
        seed: int,  
        bw_message: list,  
        tools: list = None,
        tool_choice: dict = None,  
        temperature: float = 0,  
        max_tokens: int = 500
    ):  
    try:
        time.sleep(0.5)
        # client = OpenAI()
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-07-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        response = client.chat.completions.create(  
            model=model,  
            seed=seed,  
            messages=bw_message,  
            temperature=temperature,  
            max_tokens=max_tokens,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,  
            tools=tools,  
            tool_choice=tool_choice,  
        )  

        if tool_choice is None:
            return response.choices[0].message.content
        else:
            output = response.choices[0].message.tool_calls[0].function.arguments  
  
        output_dict = json.loads(output)
        return output_dict        
          
    except openai.APIError as e:  
        print(f"\t- OpenAI API returned an API Error: {e}")  
        raise  
    except openai.APIConnectionError as e:  
        print(f"\t- OpenAI API connection error: {e}")  
        raise  
    except openai.RateLimitError as e:  
        print(f"\t- OpenAI API rate limit error: {e}")  
        raise  
    except Exception as e:  
        print(f"\t- An error occurred: {e}")  
        raise  