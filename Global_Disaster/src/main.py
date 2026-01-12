# QUY TRÌNH CHÍNH – MÔ HÌNH DỰ ĐOÁN MỨC ĐỘ NGHIÊM TRỌNG

import pandas as pd
from sklearn.model_selection import train_test_split

# ===== IMPORT CÁC MODULE =====
from src.preprocess import preprocess_data # parse_user_input không còn được sử dụng trực tiếp
from src.eda import (
    plot_target_distribution,
    plot_correlation,
    plot_top_correlated_features
)
from src.model_RFR_TuyetNhung import train_model
from src.evaluation import (
    evaluate_model,
    plot_feature_importance,
    plot_permutation_importance,
    visualize_tree
)
from src.predict import predict_from_user_input
from src.tuned_model import tune_and_train_model # Import the new tuning function

# ==== XGBoost ====
from src.model_XGBoost_LeVanTinh import train_xgb_model
from src.tuned_model_XGBoost import tune_and_train_xgb
from src.evaluation_XGBoost import evaluate_model as eval_xgb

# ===== LINEAR / RIDGE =====
from src.model_LR_DiemHanh import train_model as train_lr
from src.tuned_Ridge_DiemHanh import tune_and_train_model as tune_ridge
from src.evaluation_LR import evaluate_response_efficiency as eval_lr

def main():
    DATA_PATH = "data/global_disaster_response_2018_2024.csv"

    # BƯỚC 1: TẢI DỮ LIỆU THÔ (EDA)
    print("\n========== BƯỚC 1: TẢI DỮ LIỆU THÔ ==========")
    raw_df = pd.read_csv(DATA_PATH)
    print("Kích thước dữ liệu thô:", raw_df.shape)

    # ---------------- EDA ----------------
    print("\n========== BƯỚC 2: PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA) ==========")
    plot_target_distribution(raw_df, target="severity_index")
    plot_correlation(raw_df, target="severity_index")
    plot_top_correlated_features(raw_df, target="severity_index")

    # BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU
    print("\n========== BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU ==========")
    X, y, X_scaler, y_scaler, feature_names, y_efficiency, y_eff_scaler = preprocess_data(DATA_PATH)
    print("Kích thước X đã xử lý:", X.shape)
    print("Kích thước y mục tiêu    :", y.shape)

    # BƯỚC 4: CHIA TẬP DỮ LIỆU TRAIN – TEST
    print("\n========== BƯỚC 4: CHIA TẬP TRAIN – TEST ==========")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    X,
    y_efficiency,
    test_size=0.2,
    random_state=42
)
    print("Kích thước tập Train:", X_train.shape)
    print("Kích thước tập Test :", X_test.shape)



    # BƯỚC 5: HUẤN LUYỆN MÔ HÌNH (Sử dụng tinh chỉnh siêu tham số)
    print("\n========== BƯỚC 5: HUẤN LUYỆN RANDOM FOREST (Tuning) ==========")
    model = tune_and_train_model(X_train, y_train)
    # ===== XGBOOST =====
    print("\n========== XGBOOST ==========")
    xgb_model = tune_and_train_xgb(X_train, y_train)

    # ===== LINEAR REGRESSION =====
    print("\n========== LINEAR REGRESSION ==========")
    lr_model = train_lr(X_train_eff, y_train_eff)

    # ===== RIDGE REGRESSION =====
    print("\n========== RIDGE REGRESSION ==========")
    ridge_model = tune_ridge(X_train_eff, y_train_eff)
    print("Mô hình đã được huấn luyện với các siêu tham số tốt nhất.")

    # BƯỚC 6: ĐÁNH GIÁ
    print("\n========== BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH ==========")
    evaluate_model(model, X_test, y_test) # Regression evaluation
    print("\n===== ĐÁNH GIÁ XGBOOST ====")
    eval_xgb(xgb_model, X_test, y_test)
    print("\n===== ĐÁNH GIÁ LINEAR REGRESSION ====")
    eval_lr(lr_model, X_test_eff, y_test_eff)
    print("\n===== ĐÁNH GIÁ RIDGE REGRESSION ====")
    eval_lr(ridge_model, X_test_eff, y_test_eff)

    # BƯỚC 7: TẦM QUAN TRỌNG CỦA ĐẶC TRƯNG
    print("\n========== BƯỚC 7: TẦM QUAN TRỌNG CỦA ĐẶC TRƯNG ==========")
    fi = plot_feature_importance(model, X)
    plot_permutation_importance(model, X_test, y_test)

    print("\nTop 10 đặc trưng quan trọng:")
    print(fi.head(10))

    # BƯỚC 8: TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH
    print("\n========== BƯỚC 8: TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH ==========")
    visualize_tree(model, feature_names)

    # BƯỚC 9: NHẬP LIỆU TỪ NGƯỜI DÙNG
    print("\n========== BƯỚC 9: NHẬP LIỆU TỪ NGƯỜI DÙNG ==========")
    print("Vui lòng nhập số thực tế cho các trường sau:")

    user_input_data = {
        "response_time_hours": float(input("Thời gian phản hồi (giờ): ")),
        "aid_amount_usd": float(input("Số tiền viện trợ (USD): ")),
        "casualties": float(input("Số thương vong: ")),
        "economic_loss_usd": float(input("Thiệt hại kinh tế (USD): ")),
        "year": int(input("Năm: ")),
        "month": int(input("Tháng: "))
    }

    pred_original, level = predict_from_user_input(
        model=model,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        feature_names=feature_names,
        user_input=user_input_data # Truyền trực tiếp dữ liệu đã nhập số
    )

    print("\n===== KẾT QUẢ DỰ ĐOÁN ====")
    print(f"Chỉ số nghiêm trọng dự đoán: {pred_original:.3f}")
    print(f"Mức độ nghiêm trọng          : {level}")


if __name__ == "__main__":
    main()
