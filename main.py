from src.data_processing import load_data, explore_data,fix_data_types

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    explore_data(df)
    df = fix_data_types(df)
print("\n after the changes ")
print(df.info())