# modules/visualizer.py
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class Visualizer:
    @staticmethod
    def plot_cluster_scatter(df: pd.DataFrame, labels: np.ndarray, feature1: str, feature2: str):
        """Create 2D scatter plot of clusters using original features"""
        # Map encoded feature names back to original for display
        feature_mapping = {
            'age': 'age',
            'total_orders': 'total_orders',
            'avg_order_value': 'avg_order_value',
            'total_spend': 'total_spend',
            'last_purchase_days_ago': 'last_purchase_days_ago',
            'recency_score': 'recency_score',
            'frequency_score': 'frequency_score',
            'monetary_score': 'monetary_score',
            'clv_score': 'clv_score',
            'rfm_score': 'rfm_score'
        }
        
        # Use only original column names that exist
        actual_feature1 = feature_mapping.get(feature1, feature1)
        actual_feature2 = feature_mapping.get(feature2, feature2)
        
        # Check if features exist in dataframe
        if actual_feature1 not in df.columns:
            # Fallback to available numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            actual_feature1 = numeric_cols[0] if numeric_cols else 'total_orders'
        if actual_feature2 not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            actual_feature2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        
        fig = px.scatter(
            df, x=actual_feature1, y=actual_feature2, color=labels.astype(str),
            title=f'Customer Segments: {actual_feature1} vs {actual_feature2}',
            labels={'color': 'Segment', actual_feature1: actual_feature1.replace('_', ' ').title(),
                   actual_feature2: actual_feature2.replace('_', ' ').title()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            legend_title='Segment',
            hovermode='closest'
        )
        return fig
    
    @staticmethod
    def plot_pca_visualization(df_scaled: pd.DataFrame, labels: np.ndarray):
        """PCA visualization for high-dimensional data"""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        
        fig = px.scatter(
            x=pca_result[:, 0], y=pca_result[:, 1],
            color=labels.astype(str),
            title='PCA Visualization of Customer Segments',
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Add variance explained
        fig.update_layout(
            annotations=[
                dict(
                    x=0.95, y=0.05, xref='paper', yref='paper',
                    text=f'PC1: {pca.explained_variance_ratio_[0]:.1%}<br>PC2: {pca.explained_variance_ratio_[1]:.1%}',
                    showarrow=False, font=dict(size=10),
                    bgcolor='white', bordercolor='gray', borderwidth=1
                )
            ]
        )
        return fig
    
    @staticmethod
    def plot_cluster_radar(profile_df: pd.DataFrame):
        """Radar chart for cluster profiles"""
        # Select numeric metrics for radar chart
        metrics = ['Avg Orders', 'Avg Order Value', 'Avg Total Spend']
        
        # Check if metrics exist
        available_metrics = [m for m in metrics if m in profile_df.columns]
        
        if not available_metrics:
            return None
            
        # Normalize metrics for radar chart
        profile_norm = profile_df[available_metrics].copy()
        
        for col in available_metrics:
            min_val = profile_norm[col].min()
            max_val = profile_norm[col].max()
            if max_val > min_val:
                profile_norm[col] = (profile_norm[col] - min_val) / (max_val - min_val)
            else:
                profile_norm[col] = 0.5
        
        fig = go.Figure()
        
        for cluster in profile_df.index:
            fig.add_trace(go.Scatterpolar(
                r=profile_norm.loc[cluster, available_metrics].values,
                theta=available_metrics,
                fill='toself',
                name=f'Segment {cluster}'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Segment Profiles (Normalized)',
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_cluster_distribution(profile_df: pd.DataFrame):
        """Bar chart of cluster sizes"""
        fig = px.bar(
            profile_df, x=profile_df.index, y='Customer Count',
            title='Customer Distribution by Segment',
            labels={'index': 'Segment', 'Customer Count': 'Number of Customers'},
            color='Customer Count',
            color_continuous_scale='Blues',
            text='Customer Count'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        
        return fig
    
    @staticmethod
    def plot_feature_importance(model, feature_names: list, df_original: pd.DataFrame):
        """Plot cluster center distances (feature importance)"""
        if hasattr(model, 'cluster_centers_'):
            # Calculate standard deviation of cluster centers for each feature
            importance = np.std(model.cluster_centers_, axis=0)
            
            # Clean feature names for display
            clean_names = [name.replace('_', ' ').title() for name in feature_names]
            
            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': clean_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df, x='Importance', y='Feature',
                title='Feature Importance in Segmentation',
                labels={'Importance': 'Variance Across Segments', 'Feature': ''},
                color='Importance',
                color_continuous_scale='Viridis',
                orientation='h'
            )
            fig.update_layout(height=500)
            return fig
        return None
    
    @staticmethod
    def plot_elbow_curve(optimal_df: pd.DataFrame):
        """Plot elbow curve and silhouette scores"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=optimal_df['Clusters'], y=optimal_df['Inertia'],
            name='Inertia (Lower is Better)', 
            mode='lines+markers',
            yaxis='y1',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=optimal_df['Clusters'], y=optimal_df['Silhouette Score'],
            name='Silhouette Score (Higher is Better)', 
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Optimal Segment Selection',
            xaxis=dict(title='Number of Segments (k)', dtick=1),
            yaxis=dict(title='Inertia', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
            yaxis2=dict(title='Silhouette Score', titlefont=dict(color='red'), tickfont=dict(color='red'),
                       overlaying='y', side='right'),
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        """Plot correlation heatmap of numerical features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlations",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(height=600)
        return fig