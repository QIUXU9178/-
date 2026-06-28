import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.var_selector = VarianceThreshold(threshold=0.01)
        self.pca = PCA(n_components=0.95)  # 保留95%信息

    def clean_dirty_data(self, df):
        df_clean = df.copy()
        # 1. 删除重复样本
        df_clean = df_clean.drop_duplicates()
        # 2. 填充数值缺失：中位数
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med)
        # 3. 类别缺失填充众数
        df_clean["Class"] = df_clean["Class"].fillna(df_clean["Class"].mode()[0])
        # 4. 3σ剔除极端异常值
        for col in num_cols:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - 3*std
            upper = mean + 3*std
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean

    def feature_engineering(self, train_df, val_df, test_df):
        # 分离特征和标签
        X_train_raw = train_df.drop("Class", axis=1)
        y_train = train_df["Class"]
        X_val_raw = val_df.drop("Class", axis=1)
        y_val = val_df["Class"]
        X_test_raw = test_df.drop("Class", axis=1)
        y_test = test_df["Class"]

        # 1. 低方差特征过滤
        X_train_var = self.var_selector.fit_transform(X_train_raw)
        X_val_var = self.var_selector.transform(X_val_raw)
        X_test_var = self.var_selector.transform(X_test_raw)

        # 2. 标准化
        X_train_scaled = self.scaler.fit_transform(X_train_var)
        X_val_scaled = self.scaler.transform(X_val_var)
        X_test_scaled = self.scaler.transform(X_test_var)

        # 3. PCA降维（特征工程拓展）
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        return (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test)

    def add_noise(self, X, noise_std):
        # 给数据添加高斯噪声，用于鲁棒性测试
        noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
        return X + noise