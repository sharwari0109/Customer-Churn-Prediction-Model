# src/pipeline.py

from prefect import flow, task

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


# ---------------- TASKS ---------------- #

@task
def load_dataset(config):

    df = load_data(
        config['data']['path']
    )

    return df


@task
def explore_dataset(df):

    explore_data(df)


@task
def preprocess_data(df):

    df = fix_data_types(df)

    df = handling_missing_values(df)

    df = drop_columns(df)

    df = convert_target(df)

    return df


@task
def feature_engineering_task(df, config):

    X, y, importance = run_feature_engineering(
        df,
        config['data']['target_column']
    )

    return X, y


@task
def encoding_task(X):

    X = encode_data(X)

    return X


@task
def split_task(X, y):

    return split_data(X, y)


@task
def training_task(X_train, y_train, config):

    model = train_model(
        X_train,
        y_train,
        config
    )

    return model


@task
def evaluation_task(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nModel Performance (Test Set):\n")

    print(classification_report(
        y_test,
        y_pred
    ))


# ---------------- FLOW ---------------- #

@flow(name="Customer Churn Pipeline")

def churn_pipeline():

    # Load config
    config = load_config()

    # Load dataset
    df = load_dataset(config)

    # Explore dataset
    explore_dataset(df)

    # Preprocess
    df = preprocess_data(df)

    # Feature engineering
    X, y = feature_engineering_task(
        df,
        config
    )

    # Encoding
    X = encoding_task(X)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_task(
        X,
        y
    )

    # Train model
    model = training_task(
        X_train,
        y_train,
        config
    )

    # Evaluate model
    evaluation_task(
        model,
        X_test,
        y_test
    )


# Run pipeline
if __name__ == "__main__":

    churn_pipeline()