import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_data, load_uploaded_file, get_data_info
from modules.preprocessor import DataPreprocessor
from modules.clustering import CustomerSegmenter
from modules.visualizer import Visualizer
from utils.helpers import format_currency, get_segment_name, generate_segment_insights
import yaml
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🎯 Customer Segmentation Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio("Data Source", ["Use Sample Data", "Upload CSV File"])
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            df = load_uploaded_file(uploaded_file)
        else:
            st.info("Please upload a CSV file")
            df = None
    else:
        # Load the provided dataset
        df = load_data("data/customers.csv")
    
    if df is not None:
        st.success(f"✅ Loaded {len(df)} customers")
        
        # Clustering parameters
        st.subheader("🎯 Clustering Parameters")
        n_clusters = st.slider("Number of Segments", 2, 8, config['clustering']['n_clusters'])
        
        # Feature selection
        st.subheader("📊 Features to Use")
        use_demographics = st.checkbox("Demographics (Age, Gender)", value=True)
        use_behavioral = st.checkbox("Behavioral (Orders, Spend)", value=True)
        use_recency = st.checkbox("Recency (Last Purchase)", value=True)
        use_preferences = st.checkbox("Preferences (Category, Location)", value=True)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            scaler_option = st.selectbox("Scaler", ["standard", "minmax", "robust"], 
                                        index=["standard", "minmax", "robust"].index(config['preprocessing']['scaler']))
            handle_outliers = st.checkbox("Handle Outliers", value=config['preprocessing']['handle_outliers'])
        
        # Update config
        config['clustering']['n_clusters'] = n_clusters
        config['preprocessing']['scaler'] = scaler_option
        config['preprocessing']['handle_outliers'] = handle_outliers
        
        # Run clustering button
        run_clustering = st.button("🚀 Run Segmentation", type="primary", use_container_width=True)

# Main content
if df is not None and run_clustering:
    with st.spinner("Processing customer data and running segmentation..."):
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.preprocess(df, fit=True)
        
        # Select features based on user choice
        feature_cols = []
        if use_demographics:
            feature_cols.extend(['age', 'gender_encoded'])
        if use_behavioral:
            feature_cols.extend(['total_orders', 'avg_order_value', 'total_spend'])
        if use_recency:
            feature_cols.extend(['last_purchase_days_ago', 'recency_score'])
        if use_preferences:
            feature_cols.extend(['location_encoded', 'product_category_preference_encoded'])
        
        # Add RFM features
        feature_cols.extend(['frequency_score', 'monetary_score', 'clv_score', 'rfm_score'])
        
        # Remove duplicates
        feature_cols = list(set(feature_cols))
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
        
        X = df_processed[feature_cols]
        
        # Run clustering
        segmenter = CustomerSegmenter(n_clusters=n_clusters, random_state=config['clustering']['random_state'])
        segmenter.fit(X)
        
        # Get metrics
        metrics = segmenter.get_cluster_metrics(X)
        labels = segmenter.get_cluster_labels()
        
        # Add clusters to original dataframe
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = labels
        
        # Generate cluster profiles
        profile = segmenter.get_cluster_profiles(df)
        
        # Store in session state for persistence
        st.session_state['df'] = df_with_clusters
        st.session_state['profile'] = profile
        st.session_state['metrics'] = metrics
        st.session_state['segmenter'] = segmenter
        st.session_state['X'] = X
        st.session_state['labels'] = labels
        st.session_state['feature_cols'] = feature_cols

# Display results if clustering has been run
if 'df' in st.session_state:
    df_with_clusters = st.session_state['df']
    profile = st.session_state['profile']
    metrics = st.session_state['metrics']
    segmenter = st.session_state['segmenter']
    X = st.session_state['X']
    labels = st.session_state['labels']
    feature_cols = st.session_state['feature_cols']
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df_with_clusters))
    with col2:
        st.metric("Segments", len(profile))
    with col3:
        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
    with col4:
        st.metric("Inertia", f"{metrics['inertia']:,.0f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Segment Overview", "📈 Visualizations", "🔍 Segment Details", "📋 Customer Analysis", "💡 Insights & Actions"])
    
    with tab1:
        st.subheader("Customer Segment Profiles")
        
        # Display profile table with formatting
        display_profile = profile.copy()
        display_profile['Avg Order Value'] = display_profile['Avg Order Value'].apply(lambda x: format_currency(x))
        display_profile['Avg Total Spend'] = display_profile['Avg Total Spend'].apply(lambda x: format_currency(x))
        display_profile['Customer Count'] = display_profile['Customer Count'].astype(int)
        display_profile['Percentage'] = display_profile['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_profile, use_container_width=True)
        
        # Segment distribution chart
        st.subheader("Segment Distribution")
        fig_dist = Visualizer.plot_cluster_distribution(profile)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.subheader("Visual Analytics")
    
        # PCA Visualization
        st.markdown("### PCA Projection")
        fig_pca = Visualizer.plot_pca_visualization(X, labels)
        if fig_pca:
            st.plotly_chart(fig_pca, use_container_width=True)
    
        col1, col2 = st.columns(2)
    
        with col1:
            # Scatter plot options - use original features only
            st.markdown("### 2D Segment View")
        
            # Get only original numeric columns for plotting
            original_numeric_cols = ['total_orders', 'avg_order_value', 'total_spend', 
                                     'last_purchase_days_ago', 'age']
            available_cols = [col for col in original_numeric_cols if col in df_with_clusters.columns]
        
            if available_cols:
                x_axis = st.selectbox("X-axis", available_cols, index=0)
                y_axis = st.selectbox("Y-axis", available_cols, index=min(1, len(available_cols)-1))
            
                fig_scatter = Visualizer.plot_cluster_scatter(df_with_clusters, labels, x_axis, y_axis)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("No numeric columns available for scatter plot")
    
        with col2:
            # Radar chart
            st.markdown("### Segment Comparison")
            fig_radar = Visualizer.plot_cluster_radar(profile)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Not enough metrics for radar chart")
        
    with tab3:
        st.subheader("Detailed Segment Analysis")
        
        # Segment selector
        selected_segment = st.selectbox("Select Segment", profile.index)
        
        # Get insights for selected segment
        insights = generate_segment_insights(profile, selected_segment)
        segment_name = get_segment_name(selected_segment, profile)
        
        st.markdown(f"## {segment_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👥 Customer Count", insights['size'])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Average Spend", insights['avg_spend'])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📦 Average Orders", insights['avg_orders'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💵 Avg Order Value", insights['avg_order_value'])
        with col2:
            st.metric("⏰ Last Purchase", insights['recency'])
        with col3:
            st.metric("📍 Top Location", insights['top_location'])
        
        st.markdown(f"**Preferred Category:** {insights['top_category']}")
        st.markdown(f"**Average Age:** {insights['avg_age']}")
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"**💡 Recommendation:** {insights['recommendation']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show customers in this segment
        with st.expander(f"View Customers in {segment_name}"):
            segment_customers = df_with_clusters[df_with_clusters['Cluster'] == selected_segment]
            st.dataframe(segment_customers.head(20), use_container_width=True)
    
    with tab4:
        st.subheader("Customer-Level Analysis")
        
        # Customer search
        customer_id = st.text_input("Search by Customer ID", placeholder="e.g., CUST0001")
        
        if customer_id:
            customer_data = df_with_clusters[df_with_clusters['customer_id'].str.upper() == customer_id.upper()]
            
            if not customer_data.empty:
                cust = customer_data.iloc[0]
                cluster = int(cust['Cluster'])
                segment_name = get_segment_name(cluster, profile)
                
                st.markdown(f"## Customer: {customer_id}")
                st.markdown(f"**Segment:** {segment_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", f"{cust['age']} years")
                    st.metric("Gender", cust['gender'])
                    st.metric("Location", cust['location'])
                
                with col2:
                    st.metric("Total Orders", cust['total_orders'])
                    st.metric("Avg Order Value", format_currency(cust['avg_order_value']))
                    st.metric("Total Spend", format_currency(cust['total_spend']))
                
                with col3:
                    st.metric("Last Purchase", f"{cust['last_purchase_days_ago']} days ago")
                    st.metric("Preferred Category", cust['product_category_preference'])
                
                # Similar customers
                st.markdown("### Similar Customers")
                similar = df_with_clusters[df_with_clusters['Cluster'] == cluster].head(10)
                st.dataframe(similar[['customer_id', 'age', 'total_spend', 'total_orders', 'product_category_preference']], 
                           use_container_width=True)
            else:
                st.warning("Customer ID not found")
    
    with tab5:
        st.subheader("Business Insights & Actionable Strategies")
        
        # Overall insights
        st.markdown("### 🎯 Key Findings")
        
        # Find most valuable segment
        valuable_segment = profile['Avg Total Spend'].idxmax()
        valuable_name = get_segment_name(valuable_segment, profile)
        
        # Find most active segment
        active_segment = profile['Avg Orders'].idxmax()
        active_name = get_segment_name(active_segment, profile)
        
        # Find at-risk segment
        at_risk_segment = profile['Avg Days Since Last Purchase'].idxmax()
        at_risk_name = get_segment_name(at_risk_segment, profile)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**💎 Most Valuable:** {valuable_name}\n\nAverage spend: {format_currency(profile.loc[valuable_segment, 'Avg Total Spend'])}")
        with col2:
            st.success(f"**🔄 Most Active:** {active_name}\n\nAverage orders: {profile.loc[active_segment, 'Avg Orders']:.1f}")
        with col3:
            st.warning(f"**⚠️ At Risk:** {at_risk_name}\n\nDays since last purchase: {profile.loc[at_risk_segment, 'Avg Days Since Last Purchase']:.0f}")
        
        # Segment-wise strategies
        st.markdown("### 📋 Segment-Specific Strategies")
        
        for cluster in profile.index:
            segment_name = get_segment_name(cluster, profile)
            insights = generate_segment_insights(profile, cluster)
            
            with st.expander(f"{segment_name} - {insights['size']}"):
                st.markdown(f"**Characteristics:**")
                st.markdown(f"- Average spend: {insights['avg_spend']}")
                st.markdown(f"- Average orders: {insights['avg_orders']}")
                st.markdown(f"- Top category: {insights['top_category']}")
                st.markdown(f"- Top location: {insights['top_location']}")
                st.markdown(f"\n**Recommended Action:** {insights['recommendation']}")
        
        # Download functionality
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_segments = df_with_clusters.to_csv(index=False)
            st.download_button(
                label="Download Segmented Customers (CSV)",
                data=csv_segments,
                file_name="customer_segments.csv",
                mime="text/csv"
            )
        with col2:
            csv_profile = profile.to_csv()
            st.download_button(
                label="Download Segment Profiles (CSV)",
                data=csv_profile,
                file_name="segment_profiles.csv",
                mime="text/csv"
            )

elif df is not None and not run_clustering:
    st.info("👈 Click 'Run Segmentation' in the sidebar to start the analysis")

elif df is None:
    st.warning("⚠️ Please load data to begin")

# Footer
st.markdown("---")
st.markdown("### 🚀 Powered by K-Means Clustering | Real-time Customer Segmentation")