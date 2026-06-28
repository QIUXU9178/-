import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class Evaluator:
    def __init__(self, save_dir="../output"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.result_table = []

    def run_full_eval(self, trained_models, train_data, val_data, test_data, preprocessor):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        # 1. 无噪声基础评估
        for name, info in trained_models.items():
            res_train = self.eval_model(info, X_train, y_train)
            res_val = self.eval_model(info, X_val, y_val)
            res_test = self.eval_model(info, X_test, y_test)
            overfit_gap = res_train["accuracy"] - res_test["accuracy"]

            # 2. 不同噪声鲁棒性测试 0.1/0.3/0.5
            noise_accs = []
            for noise in [0.1, 0.3, 0.5]:
                noise_res = self.eval_model(info, X_test, y_test, noise_std=noise)
                noise_accs.append(noise_res["accuracy"])

            row = {
                "模型名称": name,
                "训练集精度": round(res_train["accuracy"],4),
                "验证集精度": round(res_val["accuracy"],4),
                "测试集精度": round(res_test["accuracy"],4),
                "过拟合差值(训练-测试)": round(overfit_gap,4),
                "单样本推理ms": round(res_test["infer_ms_per_sample"],6),
                "噪声0.1精度": round(noise_accs[0],4),
                "噪声0.3精度": round(noise_accs[1],4),
                "噪声0.5精度": round(noise_accs[2],4)
            }
            self.result_table.append(row)

        # 保存汇总表格（论文直接用）
        res_df = pd.DataFrame(self.result_table)
        res_df.to_csv(f"{self.save_dir}/model_compare.csv", index=False, encoding="utf-8-sig")
        print("\n=== 多模型对比汇总表 ===")
        print(res_df)

        # 绘图1：测试集精度柱状图
        plt.figure(figsize=(8,5))
        plt.bar(res_df["模型名称"], res_df["测试集精度"])
        plt.title("各模型测试集分类精度对比")
        plt.ylim(0.7,1.0)
        plt.ylabel("准确率")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/acc_compare.png", dpi=300)
        plt.close()

        # 绘图2：噪声鲁棒性折线图
        plt.figure(figsize=(9,5))
        noise_level = [0,0.1,0.3,0.5]
        for idx, row in res_df.iterrows():
            accs = [row["测试集精度"], row["噪声0.1精度"], row["噪声0.3精度"], row["噪声0.5精度"]]
            plt.plot(noise_level, accs, marker="o", label=row["模型名称"])
        plt.legend()
        plt.title("不同噪声强度下模型精度变化（鲁棒性）")
        plt.xlabel("高斯噪声标准差")
        plt.ylabel("准确率")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/noise_robust.png", dpi=300)
        plt.close()

        # 绘图3：过拟合差值对比
        plt.figure(figsize=(8,5))
        plt.bar(res_df["模型名称"], res_df["过拟合差值(训练-测试)"])
        plt.title("各模型过拟合程度差值（训练-测试精度）")
        plt.ylabel("精度差值，越大越容易过拟合")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/overfit_gap.png", dpi=300)
        plt.close()
        print(f"\n全部评估图表、结果表格已保存至 {self.save_dir}")

    def eval_model(self, model_info, X, y, noise_std=0):
        model = model_info["model"]
        if noise_std > 0:
            X_noise = X + np.random.normal(0, noise_std, X.shape)
            pred = model.predict(X_noise)
        else:
            pred = model.predict(X)
        acc = np.mean(pred == y)
        # 推理耗时
        import time
        t1 = time.time()
        model.predict(X[:1000])
        t_cost = (time.time()-t1)/1000 * 1000
        return {"accuracy": acc, "infer_ms_per_sample": t_cost}