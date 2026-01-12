from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

def tune_and_train_xgb(X_train, y_train):
    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        base_model,
        param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    gs.fit(X_train, y_train)

    print("Best RMSE (CV):", -gs.best_score_)
    print("Best params:", gs.best_params_)

    return gs.best_estimator_

