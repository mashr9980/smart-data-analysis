import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def is_valid_dataframe(df):
    """
    Check if dataframe is valid.
    
    Args:
        df (pd.DataFrame): Dataframe to check
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(df, pd.DataFrame) and not df.empty

def create_fallback_dataframe():
    """
    Create a fallback dataframe for when data generation fails.
    
    Returns:
        pd.DataFrame: A simple fallback dataframe
    """
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': [10, 24, 15, 32, 18]
    })

def parse_date_range(range_str):
    """
    Parse a date range string into start and end dates.
    
    Args:
        range_str (str): String describing a date range (e.g., "last 12 months")
    
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    end_date = datetime.now()
    
    if "last" in range_str.lower() and "month" in range_str.lower():
        months_match = re.search(r'(\d+)', range_str)
        months = int(months_match.group(1)) if months_match else 12
        start_date = end_date - timedelta(days=30*months)
    else:
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date

def parse_numeric_range(range_str):
    """
    Parse a numeric range string into min and max values.
    
    Args:
        range_str (str): String describing a numeric range (e.g., "1-100")
    
    Returns:
        tuple: (min_val, max_val) as numeric values
    """
    if "-" in range_str:
        numbers = list(map(int, re.findall(r'(\d+)', range_str)))
        if len(numbers) >= 2:
            min_val, max_val = numbers[0], numbers[-1]
        else:
            min_val, max_val = (0, 100) if not numbers else (0, numbers[0])
    else:
        min_val, max_val = 0, 100
    
    if min_val >= max_val:
        min_val, max_val = 0, min_val * 2 if min_val > 0 else 100
    
    return min_val, max_val

def parse_categorical_range(range_str):
    """
    Parse a categorical range string into a list of categories.
    
    Args:
        range_str (str): String describing categorical values (e.g., "A,B,C,D")
    
    Returns:
        list: List of category values
    """
    if "," in range_str:
        categories = [c.strip() for c in range_str.split(",") if c.strip()]
    else:
        if "/" in range_str:
            categories = [c.strip() for c in range_str.split("/") if c.strip()]
        elif " " in range_str:
            categories = [c.strip() for c in range_str.split() if c.strip()]
        else:
            categories = ["A", "B", "C", "D"]
    
    if not categories:
        categories = ["A", "B", "C", "D"]
    
    return categories