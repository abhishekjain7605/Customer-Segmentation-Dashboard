import pandas as pd
import streamlit as st
from typing import Optional

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load customer data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def get_data_info(df: pd.DataFrame) -> dict:
    """Get comprehensive data information"""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=['object']).columns.tolist()
    }