import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List
import streamlit as st

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.imputers = {}
        
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df_processed = df.copy()
        
        # 1. Remove duplicates if any
        df_processed = df_processed.drop_duplicates(subset=['customer_id'])
        
        # 2. Handle missing values
        df_processed = self._handle_missing_values(df_processed, fit)
        
        # 3. Encode categorical variables
        df_processed = self._encode_categorical(df_processed, fit)
        
        # 4. Create additional features (RFM-like)
        df_processed = self._create_rfm_features(df_processed)
        
        # 5. Handle outliers (optional)
        if self.config['preprocessing']['handle_outliers']:
            df_processed = self._handle_outliers(df_processed, fit)
        
        # 6. Scale numerical features
        df_processed, self.scaler = self._scale_features(df_processed, fit)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Handle missing values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median() if fit else self.imputers[col]
                df[col].fillna(median_val, inplace=True)
                if fit:
                    self.imputers[col] = median_val
        
        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if fit else self.imputers[col]
                df[col].fillna(mode_val, inplace=True)
                if fit:
                    self.imputers[col] = mode_val
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding"""
        categorical_cols = ['gender', 'location', 'product_category_preference']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col + '_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RFM-style features"""
        # Recency score (lower days = better)
        df['recency_score'] = 1 / (df['last_purchase_days_ago'] + 1)
        
        # Frequency score
        df['frequency_score'] = np.log1p(df['total_orders'])
        
        # Monetary score
        df['monetary_score'] = np.log1p(df['total_spend'])
        
        # Average order value (already exists)
        
        # Customer lifetime value indicator
        df['clv_score'] = df['total_spend'] / (df['last_purchase_days_ago'] + 1)
        
        # Combined RFM score
        df['rfm_score'] = (df['recency_score'] + df['frequency_score'] + df['monetary_score']) / 3
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Cap outliers at specified threshold"""
        numeric_cols = ['age', 'total_orders', 'avg_order_value', 'total_spend', 'last_purchase_days_ago']
        threshold = self.config['preprocessing']['outlier_threshold']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, object]:
        """Scale numerical features"""
        scaler_type = self.config['preprocessing']['scaler']
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        # Select features for scaling
        scale_cols = ['age', 'total_orders', 'avg_order_value', 'total_spend', 
                      'last_purchase_days_ago', 'recency_score', 'frequency_score', 
                      'monetary_score', 'clv_score', 'rfm_score']
        
        # Also include encoded categoricals
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        scale_cols.extend(encoded_cols)
        
        # Keep only columns that exist
        scale_cols = [col for col in scale_cols if col in df.columns]
        self.feature_names = scale_cols
        
        if fit:
            df_scaled = scaler.fit_transform(df[scale_cols])
            return pd.DataFrame(df_scaled, columns=scale_cols, index=df.index), scaler
        else:
            df_scaled = scaler.transform(df[scale_cols])
            return pd.DataFrame(df_scaled, columns=scale_cols, index=df.index), scaler