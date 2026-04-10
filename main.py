from src.data_processing import load_data, explore_data

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    explore_data(df)