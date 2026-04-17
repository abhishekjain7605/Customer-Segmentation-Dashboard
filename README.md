# Customer-Segmentation-Dashboard
Customer Segmentation Dashboard — An interactive Streamlit web app that uses K-Means clustering to segment customers into meaningful groups based on behavioral, demographic, and RFM (Recency, Frequency, Monetary) features, with rich visualizations and actionable business insights.
# 🎯 Customer Segmentation Dashboard

An interactive web application that uses **K-Means clustering** to segment customers into meaningful groups based on behavioral, demographic, and RFM (Recency, Frequency, Monetary) features — with rich visualizations and actionable business insights.

Built with **Python**, **Streamlit**, and **scikit-learn**.

---

## 📸 Features

- 📂 **Upload your own CSV** or use built-in sample data
- ⚙️ **Configurable clustering** — choose number of segments (2–8), scaler type, and features
- 🧠 **RFM Feature Engineering** — auto-generates recency, frequency, monetary, and CLV scores
- 📊 **5 interactive dashboard tabs:**
  - Segment Overview — profile table + distribution chart
  - Visualizations — PCA projection, 2D scatter, radar chart
  - Segment Details — per-segment metrics and recommendations
  - Customer Analysis — search any customer by ID
  - Insights & Actions — most valuable, most active, at-risk segments
- 📥 **Export results** — download segmented customers and segment profiles as CSV
- 📈 **Clustering quality metrics** — Silhouette Score, Calinski-Harabasz, Davies-Bouldin, Inertia

---

## 🗂️ Project Structure

```
customer-segmentation-dashboard/
│
├── app.py                  # Main Streamlit application
├── config.yaml             # Clustering and preprocessing configuration
├── requirements.txt        # Python dependencies
│
├── modules/
│   ├── data_loader.py      # CSV loading with caching
│   ├── preprocessor.py     # Preprocessing pipeline (impute, encode, scale, RFM)
│   ├── clustering.py       # K-Means segmentation + metrics
│   └── visualizer.py       # Plotly charts and visualizations
│
├── utils/
│   └── helpers.py          # Segment naming and insight generation
│
└── data/
    └── customers.csv       # Sample dataset (optional)
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
clustering:
  n_clusters: 4         # Default number of segments
  random_state: 42
  max_iter: 300

features:
  numerical:
    - age
    - total_orders
    - avg_order_value
    - total_spend
    - last_purchase_days_ago
  categorical:
    - gender
    - location
    - product_category_preference

preprocessing:
  scaler: "standard"        # Options: standard | minmax | robust
  handle_outliers: true
  outlier_threshold: 3
```

---

## 📋 Input Data Format

Your CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique customer identifier |
| `age` | int | Customer age |
| `gender` | string | Gender |
| `location` | string | City or region |
| `total_orders` | int | Number of orders placed |
| `avg_order_value` | float | Average value per order |
| `total_spend` | float | Lifetime spend |
| `last_purchase_days_ago` | int | Days since last purchase |
| `product_category_preference` | string | Preferred product category |

> Missing values are handled automatically via median (numeric) and mode (categorical) imputation.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-segmentation-dashboard.git
cd customer-segmentation-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

---

## 🧠 How It Works

```
Raw CSV
   ↓
DataPreprocessor
   ├── Remove duplicates
   ├── Impute missing values (median / mode)
   ├── Label encode categoricals
   ├── Engineer RFM features
   ├── Cap outliers (IQR-based)
   └── Scale features (Standard / MinMax / Robust)
   ↓
CustomerSegmenter (K-Means)
   ├── Fit clusters
   ├── Compute metrics (Silhouette, CH, DB, Inertia)
   └── Generate cluster profiles
   ↓
Visualizer + Helpers
   └── Charts, segment names, recommendations
   ↓
Streamlit Dashboard (5 tabs)
```

---

## 📊 Segment Types (Auto-Named)

| Segment | Characteristics |
|---------|----------------|
| 🌟 VIP Champions | High orders + High spend + Recent purchase |
| 💰 High Value (At Risk) | High orders + High spend + Not recent |
| 🔄 Frequent Shoppers | High orders + Lower spend |
| 💎 Occasional Big Spenders | Low orders + High spend |
| ❄️ Inactive Customers | Low orders + Low spend + Not recent |
| 🌱 New / Low Value Customers | Low orders + Low spend + Recent |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)
