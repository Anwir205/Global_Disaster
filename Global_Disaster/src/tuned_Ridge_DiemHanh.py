from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def tune_and_train_model(X_train, y_train):
    # Không gian tham số alpha
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }

    ridge = Ridge()

    # GridSearchCV để chọn alpha tốt nhất
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    # Huấn luyện và tìm alpha tốt nhất
    grid_search.fit(X_train, y_train)

    # Trả về mô hình tốt nhất
    return grid_search.best_estimator_
