import pandas as pd
import numpy as np

def format_currency(value: float) -> str:
    """Format currency values"""
    return f"${value:,.2f}"

def get_segment_name(cluster_id: int, profile: pd.DataFrame) -> str:
    """Generate descriptive names for segments based on characteristics"""
    row = profile.loc[cluster_id]
    
    if row['Avg Orders'] > profile['Avg Orders'].median():
        if row['Avg Total Spend'] > profile['Avg Total Spend'].median():
            if row['Avg Days Since Last Purchase'] < profile['Avg Days Since Last Purchase'].median():
                return "🌟 VIP Champions"
            else:
                return "💰 High Value (At Risk)"
        else:
            return "🔄 Frequent Shoppers"
    else:
        if row['Avg Total Spend'] > profile['Avg Total Spend'].median():
            return "💎 Occasional Big Spenders"
        else:
            if row['Avg Days Since Last Purchase'] > profile['Avg Days Since Last Purchase'].median():
                return "❄️ Inactive Customers"
            else:
                return "🌱 New/Low Value Customers"

def generate_segment_insights(profile: pd.DataFrame, cluster_id: int) -> dict:
    """Generate actionable insights for each segment"""
    row = profile.loc[cluster_id]
    
    insights = {
        'size': f"{row['Customer Count']} customers ({row['Percentage']:.1f}%)",
        'avg_spend': format_currency(row['Avg Total Spend']),
        'avg_orders': f"{row['Avg Orders']:.1f} orders",
        'avg_order_value': format_currency(row['Avg Order Value']),
        'recency': f"{row['Avg Days Since Last Purchase']:.0f} days ago",
        'top_location': row['location'],
        'top_category': row['product_category_preference'],
        'avg_age': f"{row['Avg Age']:.0f} years"
    }
    
    # Add recommendations
    if row['Avg Orders'] < 5:
        insights['recommendation'] = "Increase purchase frequency with loyalty program"
    elif row['Avg Days Since Last Purchase'] > 90:
        insights['recommendation'] = "Reactivate with special offers"
    elif row['Avg Total Spend'] > 10000:
        insights['recommendation'] = "Offer premium membership and exclusive products"
    else:
        insights['recommendation'] = "Cross-sell based on category preference"
    
    return insights