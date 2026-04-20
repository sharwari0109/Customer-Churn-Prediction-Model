from src.data_processing import (
    convert_target,
    load_data,
    explore_data,
    fix_data_types,
    handling_missing_values,
    drop_columns,
    encode_data
    
)
from src.feature_engineering import run_feature_engineering

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")

    explore_data(df)

    df = fix_data_types(df)
    df = handling_missing_values(df)
    df = drop_columns(df)
    df = convert_target(df)
    X, y, importance = run_feature_engineering(df, 'Churn')
    print("\nFinal Data Info:\n")
    df.info()
    print(df['Churn'].unique())