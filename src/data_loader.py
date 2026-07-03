import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 删掉seaborn导入
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class DataLoader:
    def __init__(self):
        # 固定绝对路径，不受运行目录影响
        base_dir = r"D:\机器学习\期末\data"
        self.train_path = os.path.join(base_dir, "Dry_Bean_Dataset_Dirty_train.csv")
        self.val_path = os.path.join(base_dir, "Dry_Bean_Dataset_Dirty_val.csv")
        self.test_path = os.path.join(base_dir, "Dry_Bean_Dataset_Dirty_test.csv")
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.all_df = None

    # 修正缩进，方法放到类内部、__init__外面
    def load_all_data(self):
        # 读取脏数据集，标记“?”为缺失值
        self.df_train = pd.read_csv(self.train_path, na_values=["?"])
        self.df_val = pd.read_csv(self.val_path, na_values=["?"])
        self.df_test = pd.read_csv(self.test_path, na_values=["?"])
        self.all_df = pd.concat([self.df_train, self.df_val, self.df_test], ignore_index=True)
        print("=== 数据集基础信息 ===")
        print(f"训练集样本：{self.df_train.shape}")
        print(f"验证集样本：{self.df_val.shape}")
        print(f"测试集样本：{self.df_test.shape}")
        print(f"全部总样本：{self.all_df.shape}")
        print("\n标签类别分布：")
        print(self.all_df["Class"].value_counts())
        return self.df_train, self.df_val, self.df_test

    def eda_analysis(self, save_path=r"D:\机器学习\期末\output\eda"):
        os.makedirs(save_path, exist_ok=True)
        df = self.all_df
        # 1. 缺失值统计
        miss_info = df.isnull().sum()
        print("\n=== 缺失值统计（脏数据）===")
        print(miss_info[miss_info > 0])
        # 2. 重复样本
        dup = df.duplicated().sum()
        print(f"\n重复样本数量：{dup}")

        # ========== 纯matplotlib替代sns.countplot 类别分布图 ==========
        plt.figure(figsize=(10,5))
        label_counts = df["Class"].value_counts()
        plt.bar(label_counts.index, label_counts.values)
        plt.title("干豆数据集类别分布")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/label_dist.png", dpi=300)
        plt.close()

        # ========== 纯matplotlib替代sns.heatmap 热力图 ==========
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(12,10))
        im = ax.imshow(corr, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        plt.title("特征相关性热力图")
        plt.tight_layout()
        plt.savefig(f"{save_path}/corr_heat.png", dpi=300)
        plt.close()

        # ========== 纯matplotlib替代sns.boxplot 箱线图 ==========
        feats = num_df.columns[:6]
        fig, axes = plt.subplots(2,3, figsize=(15,8))
        axes = axes.flatten()
        for i, feat in enumerate(feats):
            axes[i].boxplot(num_df[feat].dropna())
            axes[i].set_title(f"{feat} 异常值分布")
        plt.tight_layout()
        plt.savefig(f"{save_path}/outlier_box.png", dpi=300)
        plt.close()

        print(f"\nEDA图表已保存至 {save_path}")

if __name__ == "__main__":
    loader = DataLoader()
    tr, va, te = loader.load_all_data()
    loader.eda_analysis()