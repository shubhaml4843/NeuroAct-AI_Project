"""Data loader utility for all agents"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import json

def load_data(file_path: Union[str, Path, None]) -> Optional[pd.DataFrame]:
    """Load data from various file formats"""
    if not file_path:
        return None
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            # Try CSV as default
            return pd.read_csv(file_path)
            
    except Exception:
        return None