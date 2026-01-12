import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(path):
    """
    QUY TRÌNH TIỀN XỬ LÝ
    -------------------
    1. Tải CSV
    2. Xử lý dữ liệu ngày tháng
    3. Chọn biến mục tiêu
    4. Loại bỏ rò rỉ dữ liệu (target leakage)
    5. Mã hóa biến phân loại
    6. Chuẩn hóa biến số
    """

    # ===== 1. LOAD DATA =====
    df = pd.read_csv(path)

    # ===== 2. DATE FEATURES =====
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.drop(columns=["date"], inplace=True)

    # ===== 3. TARGETS =====
    y_raw = pd.to_numeric(df["severity_index"], errors="coerce")
    y_eff_raw = pd.to_numeric(df["response_efficiency_score"], errors="coerce")

    X_raw = df.drop(
        columns=["severity_index", "response_efficiency_score"]
    )

    # Ensure y_raw and y_eff_raw are DataFrames for concatenation
    y_raw_df = y_raw.to_frame(name="severity_index_target")
    y_eff_raw_df = y_eff_raw.to_frame(name="response_efficiency_target")

    # ===== 4. DROP NaN  ===== # This step was causing the error
    temp_df = pd.concat(
        [
            X_raw,
            y_raw_df, # Use DataFrame here
            y_eff_raw_df, # Use DataFrame here
        ],
        axis=1,
    )

    temp_df.dropna(
        subset=["severity_index_target", "response_efficiency_target"],
        inplace=True,
    )

    temp_df = temp_df.reset_index(drop=True)

    # TÁCH LẠI
    y = temp_df["severity_index_target"]
    y_efficiency = temp_df["response_efficiency_target"]
    X = temp_df.drop(
        columns=["severity_index_target", "response_efficiency_target"]
    )

    # ===== 5. SCALE TARGET (0–1) =====
    y_scaler = MinMaxScaler()
    y_eff_scaler = MinMaxScaler()

    y = pd.Series(
        y_scaler.fit_transform(y.to_frame()).flatten(),
        index=y.index,
    )

    y_efficiency = pd.Series(
        y_eff_scaler.fit_transform(y_efficiency.to_frame()).flatten(),
        index=y_efficiency.index,
    )

    # ===== 6. REMOVE TARGET LEAKAGE =====
    if "recovery_days" in X.columns:
        X = X.drop(columns=["recovery_days"])

    # ===== 7. FEATURE TYPES =====
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    # ===== 8. CATEGORICAL =====
    X_cat = pd.get_dummies(
        X[cat_cols].fillna("Unknown"),
        drop_first=True,
    )

    # ===== 9. NUMERICAL =====
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    X_scaler = StandardScaler()
    X_num = X_scaler.fit_transform(X[num_cols])

    X_num = pd.DataFrame(
        X_num,
        columns=num_cols,
        index=X.index,
    )

    # ===== 10. MERGE =====
    X_processed = pd.concat([X_num, X_cat], axis=1)

    return (
        X_processed,
        y,
        X_scaler,
        y_scaler,
        X_processed.columns,
        y_efficiency,
        y_eff_scaler,
    )
