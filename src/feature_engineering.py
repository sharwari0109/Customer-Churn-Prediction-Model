import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# RFM FEATURE 
def create_rfm_features(df):
    df['Recency'] = df['tenure']
    
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup'
    ]
    
    df['Frequency'] = df[service_cols].apply(
        lambda row: sum(row != 'No'), axis=1
    )
    
    df['Monetary'] = df['MonthlyCharges'] 
    
    return df


def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate User Engagement"""
    
    df['AvgSessionValue'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    return df


def create_complaint_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['ComplaintScore'] = (
        (df['Contract'] == 'Month-to-month').astype(int) + 
        (df['TechSupport'] == 'No').astype(int) +
        (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    )
    
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, drop_first=True)
    return df


def correlation_filter(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    
    corr_matrix = df.corr()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    
    df = df.drop(columns=to_drop)
    
    return df


def select_important_features(X: pd.DataFrame, y: pd.Series, top_n: int = 20):
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=X.columns)
    
    top_features = importances.sort_values(ascending=False).head(top_n).index
    
    return X[top_features], importances.sort_values(ascending=False)


def run_feature_engineering(df: pd.DataFrame, target_column: str):
    
    df = create_rfm_features(df)
    df = create_session_features(df)
    df = create_complaint_features(df)
    
    df = encode_features(df)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X = correlation_filter(X)
    
    X, importance = select_important_features(X, y)
    
    return X, y, importance