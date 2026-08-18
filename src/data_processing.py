import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import NUMERICAL_FEATURES, TARGET_COL

def load_data(filepath):
    return pd.read_csv(filepath)

def create_price_categories(df):
    """Creates Low, Normal, High categories based on quantiles of price."""
    df = df.copy()
    q33 = df[TARGET_COL].quantile(0.33)
    q66 = df[TARGET_COL].quantile(0.66)
    
    def categorize(price):
        if price <= q33:
            return "Low"
        elif price <= q66:
            return "Normal"
        else:
            return "High"
            
    df['price_category'] = df[TARGET_COL].apply(categorize)
    return df

def get_preprocessor():
    """Returns a scikit-learn pipeline for data imputation and scaling."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return numeric_transformer

def prepare_data(df):
    """Splits data into train and test sets."""
    df = create_price_categories(df)
    
    X = df[NUMERICAL_FEATURES]
    y_reg = df[TARGET_COL]
    y_clf = df['price_category']
    
    # Stratified split based on classification labels
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
