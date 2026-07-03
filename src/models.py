from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import time
import numpy as np

class ModelZoo:
    def __init__(self):
        # 模型1：课堂基础 逻辑回归
        # 使用可迭代的 LogisticRegression (warm_start) 来记录训练损失曲线
        self.lr = LogisticRegression(solver="saga", max_iter=1, warm_start=True)
        # 模型2：课堂基础 随机森林
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # 模型3：课外自学 XGBoost（满足作业要求：至少1种未讲授算法）
        self.xgb = XGBClassifier(n_estimators=100, random_state=42, objective="multi:softprob")
        self.model_dict = {
            "LogisticRegression": self.lr,
            "RandomForest": self.rf,
            "XGBoost(课外拓展)": self.xgb
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        # 返回额外的训练历史（loss 曲线）用于论文绘图
        training_histories = {}
        for name, model in self.model_dict.items():
            print(f"\n开始训练 {name} ...")
            t0 = time.time()
            if name == "LogisticRegression":
                # 采用多次调用 fit 的方式记录每轮训练后的 log loss
                epochs = 30
                loss_hist = []
                for ep in range(epochs):
                    model.max_iter = 1
                    model.fit(X_train, y_train)
                    try:
                        probs = model.predict_proba(X_train)
                        loss = log_loss(y_train, probs)
                    except Exception:
                        loss = np.nan
                    loss_hist.append(loss)
                train_time = time.time() - t0
                self.trained_models[name] = {"model": model, "train_time": train_time}
                training_histories[name] = loss_hist
                print(f"{name} 训练耗时：{train_time:.2f}s, 记录 {len(loss_hist)} 轮 loss")
            elif name.startswith("XGBoost"):
                # 部分 xgboost 版本的 sklearn 接口不支持直接传回训练历史，这里直接训练但不记录 loss
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                self.trained_models[name] = {"model": model, "train_time": train_time}
                training_histories[name] = []
                print(f"{name} 训练耗时：{train_time:.2f}s")
            else:
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                self.trained_models[name] = {"model": model, "train_time": train_time}
                training_histories[name] = []
                print(f"{name} 训练耗时：{train_time:.2f}s")
        return self.trained_models, training_histories

    def evaluate_single(self, model_info, X, y, noise_std=0):
        model = model_info["model"]
        if noise_std > 0:
            X_noisy = X + np.random.normal(0, noise_std, X.shape)
            pred = model.predict(X_noisy)
        else:
            pred = model.predict(X)
        acc = accuracy_score(y, pred)
        # 推理速度
        t0 = time.time()
        # 测量多次取平均以稳定测量结果
        repeat = 5
        n_samples = min(1000, X.shape[0])
        t_sum = 0.0
        for _ in range(repeat):
            t1 = time.time()
            model.predict(X[:n_samples])
            t_sum += time.time() - t1
        infer_time = (t_sum / repeat) / n_samples
        return {"accuracy": acc, "infer_ms_per_sample": infer_time*1000}