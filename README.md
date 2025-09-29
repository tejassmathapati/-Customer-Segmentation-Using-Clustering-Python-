# -Customer-Segmentation-Using-Clustering-Python-
Unsupervised learning project that segments customers into meaningful groups using clustering algorithms (K‑Means, Hierarchical, and DBSCAN) on behavioral and demographic features. The goal is to enable targeted marketing, personalization, and data‑driven customer strategy.

# Customer-Segmentation-Using-Clustering-Python

### Overview
Unsupervised customer segmentation using clustering in Python to group customers by behavior for targeting, retention, and personalization. The workflow covers EDA, RFM feature engineering, K-Means training, elbow/silhouette evaluation, PCA visualization, and business-focused segment profiling.

### Tech stack
- Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn
- Optional: plotly, pyyaml, joblib

### Data
- Inputs: transactions with columns such as customer_id, invoice_date, quantity, price; optional demographics (age, gender, country).
- Features: RFM (Recency days, Frequency purchases, Monetary spend), optional basket size and inter-purchase interval.

### Quick start
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python -m src.pipeline --config config/params.yaml
```

### Configuration (minimal)
```yaml
data:
  transactions_path: data/raw/transactions.csv
  id_col: customer_id
  date_col: invoice_date
  qty_col: quantity
  price_col: price
features:
  asof_date: 2025-09-01
  rfm: true
  log_monetary: true
  scale: standard
  pca_components: 2
model:
  algorithm: kmeans
  k_range: [2, 10]
  random_state: 42
  n_init: 10
outputs:
  artifacts_dir: artifacts
  reports_dir: reports
```

### Project structure
- src/: data.py, clean.py, features.py, model.py, visualize.py, pipeline.py
- notebooks/: 01_eda.ipynb, 02_rfm_clustering.ipynb, 03_profiling.ipynb
- data/: raw/, processed/ (gitignored)
- artifacts/: scaler, pca, model bundles
- reports/: figures, profiles, markdown summaries
- config/: params.yaml

### Core steps
- EDA: sanity checks, missingness, outliers, distributions.
- Features: compute RFM; optional log-transform monetary; standardize features.
- Modeling: K-Means with k-means++; scan k_range; hierarchical as optional check.
- Evaluation: elbow (inertia), silhouette average; PCA 2D scatter for sanity.
- Profiling: per-cluster means, counts, naming, and action recommendations.

### Example snippets
Standardize and tune K:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_scaled = StandardScaler().fit_transform(X)
scores = {k: None for k in range(2, 11)}
for k in scores:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
    scores[k] = {"inertia": km.inertia_, "silhouette": silhouette_score(X_scaled, km.labels_)}
```

Persist artifacts:
```python
import joblib
joblib.dump({"scaler": scaler, "pca": pca, "model": km}, "artifacts/segmenter.joblib")
```

### Results and actions
- Typical segments: High-Value Loyal, Recent High-Spend, At-Risk Lapsed, New/Low-Engagement.
- Actions: loyalty perks for high value, win-back sequences for at-risk, onboarding offers for new, cross-sell for frequent mid-spenders.

### Limitations
- K-Means assumes roughly spherical clusters; sensitive to scale and outliers.
- Segments drift; schedule retraining and monitor data freshness.

### Roadmap
- Add gap statistic and auto-K selection.
- Support mixed-type clustering (e.g., k-prototypes) and DBSCAN for density patterns.

### Contributing
Open an issue or PR with clear description; include small synthetic tests for features and model steps.
