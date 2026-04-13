import pandas as pd 

def load_data(path):
    return pd.read_csv(path)

def explore_data(df):
    print("HEAD")
    print(df.head())
    
    print("INFO")
    df.info()
    
    print("DESCRIPTION")
    print(df.describe())
    print(df.nunique())
    
def fix_data_types(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def handling_missing_values(df):
    print("Before removing missing values:\n")
    print(df.isnull().sum())

    df = df.dropna()

    print("\nAfter removing missing values:\n")
    print(df.isnull().sum())

    return df

def drop_columns(df):
    df = df.drop(columns=['customerID'])
    return df

def convert_target(df):
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_data(df):
    df = pd.get_dummies(df,drop_first=True)
    return df