import pandas as pd 

def load_data(path):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print("HEAD")
    print(df.head())
    
    print("INFO")
    print(df.info())
    print("DESCRIPTION")
    print(df.describe())
    print(df.nunique())
    
def fix_data_types(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
    return df
    