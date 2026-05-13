# main.py

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
from src.model_training import train_model
from src.train import split_data
from src.config_loader import load_config
from sklearn.metrics import classification_report


if __name__ == "__main__":

    # Load config
    config = load_config()

    # Load data
    df = load_data(
        config['data']['path']
    )

    # Explore
    explore_data(df)

    # Preprocessing
    df = fix_data_types(df)

    df = handling_missing_values(df)

    df = drop_columns(df)

    df = convert_target(df)

    # Feature Engineering
    X, y, importance = run_feature_engineering(
        df,
        config['data']['target_column']
    )

    # Encoding
    X = encode_data(X)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y
    )

    # Train model
    model = train_model(
        X_train,
        y_train,
        config
    )

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nModel Performance (Test Set):\n")

    print(classification_report(
        y_test,
        y_pred
    ))

    # Final info
    print("\nFinal Data Info:\n")

    df.info()

    print(df['Churn'].unique())