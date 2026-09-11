# Customer Personality Segmentation

An end-to-end unsupervised machine learning project that segments retail customers based on their demographics, spending habits, and purchasing behavior. The pipeline covers data cleaning, exploratory data analysis, feature engineering, preprocessing, and clustering (KMeans with and without PCA, and DBSCAN), evaluated with the Elbow Method, Silhouette Score, and the Davies-Bouldin Index — closing with a business-oriented profile of the resulting customer segments.

## Overview

The goal of this project is to help a retail/marketing business understand its customer base by grouping customers into distinct segments — e.g. high-income high-spenders, deal-driven shoppers, or low-engagement newer customers — without relying on any pre-labeled target. The analysis is built around the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) dataset (2,240 customers, 29 raw columns spanning demographics, product spending, purchase channels, and marketing campaign responses).

The project is organized as a five-stage notebook pipeline, with each stage reading the output of the previous one and writing its own output forward, so any notebook can be re-run independently once the earlier stages have executed.

## Pipeline

| Stage | Notebook | Input | Output | What it does |
|---|---|---|---|---|
| 1 | `01_data_cleaning.ipynb` | `data/raw/customer_segmentation.csv` | `data/processed/01_cleaned.csv` | Loads and inspects the raw data, drops the 24 rows with missing `Income`, and removes non-informative columns (`ID`, and the constant columns `Z_Revenue` / `Z_CostContact`) |
| 2 | `02_eda_raw.ipynb` | `data/processed/01_cleaned.csv` | — | Exploratory analysis of the cleaned-but-unengineered data: distributions of all numeric columns, boxplots for outlier inspection, and category counts for `Education` and `Marital_Status` |
| 3 | `03_feature_engineering.ipynb` | `data/processed/01_cleaned.csv` | `data/processed/03_features.csv` | Outlier removal, feature engineering, post-engineering EDA, and correlation analysis (details below) |
| 4 | `04_preprocessing_pipeline.ipynb` | `data/processed/03_features.csv` | `data/processed/04_processed.csv` | Fits a scikit-learn `ColumnTransformer` that log-transforms/scales skewed features, scales normal features, ordinally encodes `Education`, and one-hot encodes `Marital_Status` |
| 5 | `05_modeling_clustering.ipynb` | `data/processed/03_features.csv`, `data/processed/04_processed.csv` | — | Clustering experiments (KMeans + PCA, KMeans without PCA, DBSCAN), model comparison, and cluster profiling |

## Step-by-Step Details

### 1. Data Cleaning
- Raw shape: 2,240 rows × 29 columns.
- `Income` was the only column with missing values (24 rows); these rows were dropped rather than imputed, since they represent a small fraction of the dataset.
- `Z_Revenue` and `Z_CostContact` are constant across every row and were dropped as non-informative; `ID` was dropped as a unique identifier with no predictive value.
- Result: 2,216 rows × 26 columns, saved to `01_cleaned.csv`.

### 2. Exploratory Data Analysis (Raw)
- Histogram + KDE plots for every numeric column, and boxplots to flag outliers.
- Count plots for the categorical columns `Education` and `Marital_Status`.
- Key findings: several spending/purchase-count columns are right-skewed (expected for transactional data); `Age` (derived from `Year_Birth`) and `Income` contain a small number of extreme outliers; both categorical columns are unbalanced with a few rare categories.

### 3. Feature Engineering
- **Outlier handling:** customers older than 90 (using a reference year of 2015) and with `Income` ≥ 120,000 were removed, bringing the dataset to 2,198 rows.
- **Derived features:**
  - `age` (from `Year_Birth`) and an `age_group` bucket (used for EDA only, later dropped).
  - `total_children` = `Kidhome` + `Teenhome`.
  - `total_spending` = sum of all 6 product spending columns (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`).
  - `total_purchases` = sum of the 4 purchase-channel columns.
  - Per-category **spending shares** (`meat_share`, `fruit_share`, `fish_share`, `sweet_share`, `gold_share`) and per-channel **purchase shares** (`deals_share`, `web_share`, `catalog_share`, `store_share`), each normalized by the relevant total.
  - `Total_Accepted_Cmp` and `Accepted_Any_Campaign`, summarizing the 5 campaign-acceptance columns plus `Response`.
  - `Customer_Days`, the customer's tenure in days relative to the most recent enrollment date in the dataset.
- **Post-engineering EDA:** average spending, number of children, and campaign-acceptance rate by education level; spending by age group and marital status; campaign responders vs. non-responders; average basket share per product category and purchase channel; income vs. total spending colored by number of children.
- **Dropped redundant columns:** the raw spending, purchase-channel, campaign, `Teenhome`, `Kidhome`, `Dt_Customer`, and `Year_Birth` columns (now represented by the engineered features above), plus `age_group` (EDA-only). Three malformed `Marital_Status` values (`Absurd`, `YOLO`, `Alone`) were also filtered out.
- **Correlation analysis:** a heatmap of the final numeric feature set was inspected to check for redundant/highly correlated features.
- Result: 2,198 rows × 21 columns, saved to `03_features.csv`.

### 4. Preprocessing Pipeline
A single `scikit-learn` `ColumnTransformer` (defined in `src/utils.py` as `build_preprocessor`) applies:
- **Skewed numeric features** (e.g. `NumDealsPurchases`, `Complain`, `total_spending`, and the spending/purchase share columns): log-transform + scaling.
- **Normally-distributed numeric features** (e.g. `Income`, `Recency`, `Customer_Days`): standard scaling.
- **`Education`**: ordinal encoding (it has a natural order: Basic → 2n Cycle → Graduation → Master → PhD) followed by scaling.
- **`Marital_Status`**: one-hot encoding.
- Result: a fully numeric feature matrix of 2,198 rows × 25 columns, saved to `04_processed.csv` and used directly for clustering.

### 5. Modeling & Clustering
Three clustering experiments were run on the processed feature matrix:

1. **KMeans + PCA** — PCA was fit on the full feature matrix and the number of components was chosen to retain 90% of the cumulative explained variance; KMeans was then run on the reduced space for `k = 2..10`.
2. **KMeans (no PCA)** — KMeans run directly on all 25 features for `k = 2..10`.
3. **DBSCAN (no PCA)** — `eps` was estimated with a k-distance graph (10th nearest neighbor), followed by a small grid search over `eps` and `min_samples ∈ {5, 10, 15, 20}`, selecting the combination with the best Silhouette Score among valid results (2–10 clusters, noise ratio < 25%).

**Evaluation metrics** used throughout: the Elbow Method (inertia, for KMeans), Silhouette Score, and the Davies-Bouldin Index. For each approach, the best `k` (or `eps`/`min_samples`) was chosen primarily by Silhouette Score, cross-checked against the Elbow curve and the Davies-Bouldin Index.

**Model comparison:** all three approaches are summarized in a single comparison table/chart (number of clusters, Silhouette Score, Davies-Bouldin Index, noise ratio), and the model with the best Silhouette Score is selected as the final segmentation.

**Cluster profiling:** the winning segmentation is mapped back onto the original (pre-scaling) feature values — income, age, total spending, total purchases, number of children, recency, tenure, and purchase-channel shares — and visualized as a standardized heatmap, so each segment can be described in business terms.

**General trade-offs observed between the three approaches:**
- *KMeans + PCA* tends to give visually cleaner separation, since PCA removes redundancy/noise between correlated features, at some cost to fine-grained detail.
- *KMeans (no PCA)* keeps all information, but in a 25-dimensional space distances become less discriminative (curse of dimensionality), which can lower the Silhouette Score.
- *DBSCAN* handles non-spherical clusters and explicit outlier detection well, but is sensitive to `eps` and tends to flag a large share of points as noise in high dimensions, which limits a direct comparison with KMeans.

## Project Structure

```
customer-segmentation/
├── data/
│   ├── raw/                      # place customer_segmentation.csv here
│   └── processed/                 # intermediate outputs written by the notebooks
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_raw.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing_pipeline.ipynb
│   └── 05_modeling_clustering.ipynb
├── src/
│   └── utils.py                   # shared preprocessing / clustering-evaluation helpers
├── outputs/
│   └── figures/                   # optional exported figures
├── requirements.txt
└── README.md
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the dataset at data/raw/customer_segmentation.csv
jupyter notebook notebooks/
```

Run the notebooks in order, `01` through `05`. Each notebook persists its output to `data/processed/`, so later stages can be re-run independently once the earlier ones have executed at least once.

## Dataset

The raw CSV is not included in this repository. Download the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) dataset and place `customer_segmentation.csv` inside `data/raw/` before running the first notebook.

## Key Techniques Used

- Data cleaning: missing-value handling, duplicate checks, dropping non-informative columns
- Exploratory data analysis: distribution plots, boxplots, correlation heatmaps
- Feature engineering: aggregate features, ratio/share features, tenure features, outlier removal
- Preprocessing: log-transformation, standard scaling, ordinal encoding, one-hot encoding via `ColumnTransformer`
- Dimensionality reduction: PCA (variance-based component selection)
- Clustering: KMeans, DBSCAN
- Model evaluation: Elbow Method, Silhouette Score, Davies-Bouldin Index
- Cluster profiling and business interpretation

## Limitations & Possible Extensions

- The number of clusters was chosen primarily by Silhouette Score and should be cross-checked visually against the Elbow curve before being used in a business report.
- DBSCAN's `eps` and `min_samples` were chosen via a limited grid search; a wider search or a more advanced method such as OPTICS could improve results.
- For production use, additional validation is recommended — e.g. checking cluster stability across different random seeds or across different time periods of data.
