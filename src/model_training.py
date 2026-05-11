from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(
        X_train,
        y_train
    )

    print("Before SMOTE:\n", y_train.value_counts())
    print("After SMOTE:\n", y_train_res.value_counts())

    # Base model
    model = GradientBoostingClassifier(random_state=42)

    # Parameters to test
    param_grid = {

        'n_estimators': [50, 100],

        'learning_rate': [0.05, 0.1],

        'max_depth': [2, 3]
    }

    # Grid Search
    grid_search = GridSearchCV(

        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Training
    grid_search.fit(X_train_res, y_train_res)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest F1 Score:")
    print(grid_search.best_score_)

    # Return best model
    return grid_search.best_estimator_