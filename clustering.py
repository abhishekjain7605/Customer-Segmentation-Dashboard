from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
import streamlit as st

class CustomerSegmenter:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels = None
        
    def fit(self, X: pd.DataFrame) -> 'CustomerSegmenter':
        """Fit K-Means clustering"""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.labels = self.model.fit_predict(X)
        return self
    
    def get_cluster_metrics(self, X: pd.DataFrame) -> dict:
        """Calculate clustering metrics"""
        metrics = {
            'silhouette_score': silhouette_score(X, self.labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, self.labels),
            'davies_bouldin_score': davies_bouldin_score(X, self.labels),
            'inertia': self.model.inertia_
        }
        return metrics
    
    def get_cluster_profiles(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Generate cluster profiles with original features"""
        df_with_clusters = original_df.copy()
        df_with_clusters['Cluster'] = self.labels
        
        # Aggregate metrics per cluster
        profile = df_with_clusters.groupby('Cluster').agg({
            'customer_id': 'count',
            'age': 'mean',
            'total_orders': 'mean',
            'avg_order_value': 'mean',
            'total_spend': 'mean',
            'last_purchase_days_ago': 'mean',
            'gender': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
            'location': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
            'product_category_preference': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).rename(columns={
            'customer_id': 'Customer Count',
            'age': 'Avg Age',
            'total_orders': 'Avg Orders',
            'avg_order_value': 'Avg Order Value',
            'total_spend': 'Avg Total Spend',
            'last_purchase_days_ago': 'Avg Days Since Last Purchase'
        })
        
        # Add percentage
        profile['Percentage'] = (profile['Customer Count'] / profile['Customer Count'].sum()) * 100
        
        return profile
    
    def get_cluster_labels(self) -> np.ndarray:
        """Return cluster labels"""
        return self.labels
    
    @staticmethod
    def determine_optimal_clusters(X: pd.DataFrame, max_clusters: int = 10) -> pd.DataFrame:
        """Elbow method to determine optimal clusters"""
        inertias = []
        sil_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(X, labels))
        
        results = pd.DataFrame({
            'Clusters': range(2, max_clusters + 1),
            'Inertia': inertias,
            'Silhouette Score': sil_scores
        })
        
        return results