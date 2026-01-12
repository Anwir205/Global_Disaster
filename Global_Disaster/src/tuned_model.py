from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def tune_and_train_model(X_train, y_train):
    # Định nghĩa không gian tham số cho RandomizedSearchCV
    param_distributions = {
        'n_estimators': [50, 100], # Giảm mạnh phạm vi n_estimators
        'max_depth': [5, 8], # Giảm mạnh phạm vi max_depth
        'min_samples_split': [5, 10], # Giảm phạm vi min_samples_split
        'min_samples_leaf': [1, 2], # Giảm phạm vi min_samples_leaf
        'bootstrap': [True, False]
    }

    # Khởi tạo mô hình RandomForestRegressor cơ sở
    base_model = RandomForestRegressor(random_state=42)

    # Thiết lập RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10, # Số lượng kết hợp tham số khác nhau để thử
        cv=3, # Số lần cross-validation
        scoring='neg_mean_squared_error', # Tiêu chí để đánh giá mô hình
        random_state=42, # Để đảm bảo tính tái lập
        n_jobs=-1, # Sử dụng tất cả các lõi CPU có sẵn
        verbose=1 # Hiển thị tiến độ
    )

    # Thực hiện tìm kiếm ngẫu nhiên và huấn luyện mô hình
    random_search.fit(X_train, y_train)

    # Trả về mô hình tốt nhất tìm được
    return random_search.best_estimator_
