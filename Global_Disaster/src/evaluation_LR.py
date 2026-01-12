import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_response_efficiency(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n----- Đánh giá mô hình cho Response Efficiency Score -----")
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("R2  :", r2_score(y_test, y_pred))

    # Scatter plot Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Response Efficiency Score thực tế")
    plt.ylabel("Response Efficiency Score dự đoán")
    plt.title("Response Efficiency Score: Thực tế vs Dự đoán")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color='red', linestyle='--', label="Perfect Prediction")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
