from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train, config):

    # SMOTE
    smote = SMOTE(
        random_state=config['smote']['random_state']
    )

    X_train_res, y_train_res = smote.fit_resample(
        X_train,
        y_train
    )

    print("Before SMOTE:\n", y_train.value_counts())

    print("After SMOTE:\n", y_train_res.value_counts())

    # Base model
    model = GradientBoostingClassifier(
        random_state=config['model']['random_state']
    )

    # Parameter grid
    param_grid = {

        'n_estimators': [
            config['model']['n_estimators']
        ],

        'learning_rate': [
            config['model']['learning_rate']
        ],

        'max_depth': [
            config['model']['max_depth']
        ]
    }

    # Grid Search
    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=config['grid_search']['cv'],

        scoring=config['grid_search']['scoring'],

        n_jobs=-1
    )

    # Train
    grid_search.fit(X_train_res, y_train_res)

    print("\nBest Parameters:")

    print(grid_search.best_params_)

    return grid_search.best_estimator_