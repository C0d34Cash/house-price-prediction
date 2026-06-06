# House Price Prediction - Ames Housing

Predicting house prices using Linear Regression with 5-Fold Cross Validation on the Ames Housing dataset from Kaggle.

## Key Results
- **CV RMSE**: $25,492 
- **Model**: LinearRegression + StandardScaler in Pipeline + log1p target transform
- **Features**: 264 features after one-hot encoding categorical + numerical data
- **Dataset**: 1,460 houses, 81 original columns from Ames, Iowa

## How to Run This Project
1. Clone repo: `git clone https://github.com/C0d34Cash/house-price-prediction`
2. Install dependencies: `pip install -r requirements.txt`
3. Download `train.csv` + `test.csv` from [Kaggle Ames Housing competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
4. Place both CSV files in the repo folder
5. Open `notebook.ipynb` in Jupyter Notebook and run all cells

## Project Structure
house-price-prediction/
├── notebook.ipynb     # EDA, feature engineering, log transform, CV + predictions
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores data files, cache, outputs
└── LICENSE            # MIT License 

## Key Steps in Notebook
1. **Feature Engineering**: Created TotalSF, TotalBath, Age, RemodAge
2. **Target Transform**: log1p(SalePrice) to reduce right skew in house prices
3. **Missing Values**: Neighborhood median for LotFrontage, YearBuilt for GarageYrBlt, 'None' for categoricals
4. **Encoding**: One-hot encoding with drop_first=True → 264 features
5. **Evaluation**: 5-Fold CV with custom RMSE in dollars. Median $25,492 vs Mean $49,467 due to outlier fold 3
6. **Pipeline**: StandardScaler + LinearRegression to prevent data leakage between folds

## What I Learned
- Why log transform matters for skewed regression targets
- Median > Mean for CV when outliers exist in folds
- Pipelines prevent data leakage: scaler fits only on train folds
- Handling high-dimensional data: 264 features from 79 original columns

## Dataset
Kaggle - House Prices: Advanced Regression Techniques  
Kaggle Username: TH3_FR33L@NC3r

## CV Score
Median RMSE $25,492 across 5 folds. Fold 3 had outliers at $151,706 which inflated the mean.
