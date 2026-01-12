import pandas as pd
import numpy as np

def predict_from_user_input(model, X_scaler, y_scaler, feature_names, user_input): # Cập nhật chữ ký hàm
    X_user = pd.DataFrame([user_input])

    # Điền giá trị thiếu (nếu có)
    for col in feature_names:
        if col not in X_user:
            X_user[col] = 0

    X_user = X_user[feature_names]

    # ===== CHUẨN HÓA CHỈ CÁC CỘT DỮ LIỆU SỐ =====
    num_cols = X_scaler.feature_names_in_ # Sử dụng X_scaler
    X_user[num_cols] = X_scaler.transform(X_user[num_cols]) # Sử dụng X_scaler

    y_pred_scaled = model.predict(X_user)[0] # Mô hình dự đoán giá trị y đã được chuẩn hóa

    if y_pred_scaled < 0.4: # Áp dụng ngưỡng cho dự đoán đã chuẩn hóa
        level = "LOW"
    elif y_pred_scaled < 0.7:
        level = "MEDIUM"
    else:
        level = "HIGH"

    y_pred_original = y_scaler.inverse_transform([[y_pred_scaled]])[0][0] # Đảo ngược chuẩn hóa để hiển thị giá trị gốc

    return y_pred_original, level # Trả về giá trị gốc và mức độ
