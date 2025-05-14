import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from utils.api import call_llama, extract_json_from_response
from utils.data_utils import (
    parse_date_range, 
    parse_numeric_range, 
    parse_categorical_range, 
    create_fallback_dataframe
)
from config.settings import DEFAULT_SAMPLE_SIZE

def get_data_for_query(query):
    """
    Get or generate appropriate data for the given query.
    
    Args:
        query (str): User's data query
    
    Returns:
        tuple: (data_df, source_info, is_synthetic)
    """
    data_info = identify_data_needs(query)
    
    data_df = None
    source_info = "Data not available from standard sources"
    is_synthetic = True
    
    if data_df is None:
        data_df, source_info = generate_synthetic_data(query, data_info)
        is_synthetic = True
    
    return data_df, source_info, is_synthetic

def identify_data_needs(query):
    """
    Identify the data needs based on the query.
    
    Args:
        query (str): User's data query
    
    Returns:
        dict: Information about the data needs
    """
    data_identification_prompt = f"""
    Given this query: "{query}"
    
    1. What specific dataset would best answer this query?
    2. What data columns/fields would be needed?
    3. Is this likely standard data that could be retrieved from a public API?
    4. If retrievable, what would be the best source?
    
    Format your response as JSON with these exact keys: 
    {{
        "dataset_name": "name of dataset",
        "required_fields": ["field1", "field2", ...],
        "retrievable": true or false,
        "best_source": "source name if applicable"
    }}
    """
    
    try:
        data_info_response = call_llama(data_identification_prompt)
        data_info = extract_json_from_response(data_info_response)
        
        if not data_info:
            data_info = {
                "dataset_name": "unknown",
                "required_fields": [],
                "retrievable": False,
                "best_source": "none"
            }
    except Exception as e:
        data_info = {
            "dataset_name": "unknown",
            "required_fields": [],
            "retrievable": False,
            "best_source": "none"
        }
    
    return data_info

def generate_synthetic_data(query, data_info):
    """
    Generate synthetic data based on the query and data info.
    
    Args:
        query (str): User's data query
        data_info (dict): Information about the data needs
    
    Returns:
        tuple: (data_df, source_info)
    """
    synthetic_data_prompt = f"""
    Given this query: "{query}"
    
    Generate realistic synthetic data that would help answer this query. The data should look plausible and realistic.
    
    Required dataset: {data_info['dataset_name']}
    Required fields: {', '.join(data_info['required_fields']) if data_info['required_fields'] else 'To be determined'}
    
    Create a design for a small but representative dataset with:
    1. The exact column names (3-6 columns)
    2. The data types for each column
    3. The range/distribution of values for each column
    4. Any patterns, trends, or relationships that should exist in the data
    5. Recommended sample size (10-50 rows)
    
    Format your response as JSON with these exact keys:
    {{
        "columns": [
            {{"name": "column1", "type": "numeric/categorical/date", "range": "description of values"}},
            ...
        ],
        "sample_size": number,
        "patterns": ["description of pattern1", ...],
        "time_period": "if applicable"
    }}
    
    Just provide the JSON, no other text.
    """
    
    try:
        design_response = call_llama(synthetic_data_prompt)
        design = extract_json_from_response(design_response)
        
        if not design or not isinstance(design.get("columns"), list) or len(design.get("columns", [])) == 0:
            design = {
                "columns": [
                    {"name": "Date", "type": "date", "range": "last 12 months"},
                    {"name": "Value", "type": "numeric", "range": "1-100"},
                    {"name": "Category", "type": "categorical", "range": "A,B,C,D"}
                ],
                "sample_size": DEFAULT_SAMPLE_SIZE,
                "patterns": ["Upward trend over time"],
                "time_period": "Last 12 months"
            }
    except Exception as e:
        design = {
            "columns": [
                {"name": "Date", "type": "date", "range": "last 12 months"},
                {"name": "Value", "type": "numeric", "range": "1-100"},
                {"name": "Category", "type": "categorical", "range": "A,B,C,D"}
            ],
            "sample_size": DEFAULT_SAMPLE_SIZE,
            "patterns": ["Upward trend over time"],
            "time_period": "Last 12 months"
        }
    
    if not isinstance(design.get("sample_size"), int) or design["sample_size"] <= 0:
        design["sample_size"] = DEFAULT_SAMPLE_SIZE
    
    if not isinstance(design.get("patterns"), list):
        design["patterns"] = ["Random distribution"]
    
    data = {}
    for col in design["columns"]:
        if not all(k in col for k in ["name", "type", "range"]):
            continue
            
        if col["type"] == "date":
            try:
                start_date, end_date = parse_date_range(col["range"])
                date_range = pd.date_range(start=start_date, end=end_date, periods=design["sample_size"])
                data[col["name"]] = date_range
            except Exception as e:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                date_range = pd.date_range(start=start_date, end=end_date, periods=design["sample_size"])
                data[col["name"]] = date_range
            
        elif col["type"] == "numeric":
            try:
                min_val, max_val = parse_numeric_range(col["range"])
                    
                has_trend = any("trend" in p.lower() for p in design["patterns"])
                
                if has_trend:
                    if "upward" in ' '.join(design["patterns"]).lower():
                        base = np.linspace(min_val, max_val, design["sample_size"])
                    elif "downward" in ' '.join(design["patterns"]).lower():
                        base = np.linspace(max_val, min_val, design["sample_size"])
                    else:
                        base = np.sin(np.linspace(0, 4*np.pi, design["sample_size"])) * (max_val-min_val)/2 + (max_val+min_val)/2
                    
                    noise = np.random.normal(0, (max_val-min_val)/10, design["sample_size"])
                    data[col["name"]] = base + noise
                else:
                    data[col["name"]] = np.random.uniform(min_val, max_val, design["sample_size"])
            except Exception as e:
                data[col["name"]] = np.random.uniform(0, 100, design["sample_size"])
            
        elif col["type"] == "categorical":
            try:
                categories = parse_categorical_range(col["range"])
                data[col["name"]] = np.random.choice(categories, design["sample_size"])
            except Exception as e:
                data[col["name"]] = np.random.choice(["A", "B", "C", "D"], design["sample_size"])
    
    df = pd.DataFrame(data)
    
    source_info = (
        f"Synthetic data generated based on query analysis. "
        f"This simulated dataset includes {design['sample_size']} records "
        f"with {len(design['columns'])} variables. "
        f"Note: This is generated data for illustration purposes."
    )
    
    return df, source_info