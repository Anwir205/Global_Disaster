from xgboost import XGBRegressor

def train_xgb_model(X_train, y_train):
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

