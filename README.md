# Skincare Product Recommender
### End-to-End Machine Learning System

## Problem Statement
Recommend top skincare products based on skin concern and budget using Machine Learning.

## Dataset
Sephora Products and Skincare Reviews — Kaggle
- 8,216 skincare products
- 9 skin concerns (Acne, Dryness, Anti-Aging, Dark spots, Pores, Redness etc.)

## ML Pipeline
| Module | Techniques |
|---|---|
| Data Prep | Missing value handling, Label Encoding, Feature Engineering, EDA |
| Baseline Model | Logistic Regression with class balancing |
| Optimization | L1/L2 Regularization, KMeans Clustering (K=4), PCA |
| Advanced Models | Decision Tree, Random Forest, XGBoost (best model) |

## Best Model
XGBoost (Tuned) — selected via GridSearchCV

## How to Run

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### Run the notebook
Open `final_skincare.ipynb` in VS Code or Jupyter and run all cells.

### Run the web app
```bash
streamlit run app.py
```

## How to Use the Recommender
```python
recommend_skincare(concern='Acne/Blemishes', max_price=30, top_n=5)
```

Available concerns: Acne/Blemishes, Anti-Aging, Dark Circles, Dark spots, Dryness, Dullness/Uneven Texture, Loss of firmness, Pores, Redness

## Note on Dataset
Download `product_info.csv` from [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) and place it in the project folder.

## Tools Used
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit
