import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance

def evaluate_model(model, X_test, y_test):
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    print("\n----- Các chỉ số hồi quy -----")
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("R2  :", r2_score(y_test, y_pred))

    # Vẽ biểu đồ Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Thực tế")
    plt.ylabel("Dự đoán")
    plt.title("Thực tế so với Dự đoán")
    plt.show()


def plot_feature_importance(model, X):
    # Tính toán và hiển thị tầm quan trọng của các đặc trưng (dựa trên Gini)
    fi = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    fi.head(15).plot(kind="barh", figsize=(8,6))
    plt.gca().invert_yaxis()
    plt.title("Tầm quan trọng của đặc trưng (Gini)")
    plt.show()

    return fi


def plot_permutation_importance(model, X, y):
    # Tính toán và hiển thị tầm quan trọng của các đặc trưng (dựa trên Permutation)
    result = permutation_importance(
        model, X, y,
        n_repeats=10, # Số lần lặp lại
        random_state=42, # Để đảm bảo tính tái lập
        n_jobs=-1 # Sử dụng tất cả các lõi CPU có sẵn
    )

    pi = pd.Series(
        result.importances_mean,
        index=X.columns
    ).sort_values(ascending=False)

    pi.head(15).plot(kind="barh", figsize=(8,6))
    plt.gca().invert_yaxis()
    plt.title("Tầm quan trọng của Permutation")
    plt.show()

    return pi


def visualize_tree(model, feature_names):
    # Trực quan hóa một cây quyết định trong Random Forest
    plt.figure(figsize=(28,12))
    plot_tree(
        model.estimators_[0], # Cây đầu tiên trong Random Forest
        feature_names=feature_names,
        filled=True,
        max_depth=3 # Hiển thị tối đa 3 cấp độ
    )
    plt.title("Một cây quyết định trong Random Forest")
    plt.show()
