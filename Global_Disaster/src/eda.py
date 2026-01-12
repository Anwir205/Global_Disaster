import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# 1. PHÂN BỐ TARGET
def plot_target_distribution(df, target="severity_index"):
    """
    Xem phân bố của chỉ số nghiêm trọng (severity_index)
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df[target], bins=30, kde=True)
    plt.title("Phân bố chỉ số nghiêm trọng")
    plt.xlabel("Chỉ số nghiêm trọng")
    plt.ylabel("Số lượng")
    plt.tight_layout()
    plt.show()


# 2. HEATMAP TƯƠNG QUAN (NUMERICAL)
def plot_correlation(df, target="severity_index"):
    """
    Heatmap tương quan giữa các biến số
    """
    num_df = df.select_dtypes(exclude="object")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        num_df.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Heatmap tương quan (Đặc trưng số)")
    plt.tight_layout()
    plt.show()


# 3. TOP FEATURES VS TARGET
def plot_top_correlated_features(df, target="severity_index", top_k=6):
    """
    Vẽ biểu đồ scatter giữa biến mục tiêu và các biến có tương quan mạnh nhất
    """
    num_df = df.select_dtypes(exclude="object")

    corr = num_df.corr()[target].abs().sort_values(ascending=False)
    top_features = corr.index[1:top_k+1]

    for col in top_features:
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=df[col], y=df[target], alpha=0.3)
        plt.title(f"{col} vs {target}")
        plt.xlabel(col)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()

        #target RESPONSE EFFICIENCY SCORE
def plot_response_efficiency_correlation(X, y_efficiency):

    # Gộp X và y để tính tương quan
    df_combined = pd.concat([X, y_efficiency], axis=1)

    # Chỉ lấy các cột số
    numeric_cols = df_combined.select_dtypes(include="number")

    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        "Biểu đồ nhiệt tương quan giữa các đặc trưng số và Response Efficiency Score",
        fontsize=14
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
