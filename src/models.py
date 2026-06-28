from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

class ModelZoo:
    def __init__(self):
        # 模型1：课堂基础 逻辑回归
        self.lr = LogisticRegression(max_iter=1000)
        # 模型2：课堂基础 随机森林
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # 模型3：课外自学 XGBoost（满足作业要求：至少1种未讲授算法）
        self.xgb = XGBClassifier(n_estimators=100, random_state=42, objective="multi:softmax")
        self.model_dict = {
            "LogisticRegression": self.lr,
            "RandomForest": self.rf,
            "XGBoost(课外拓展)": self.xgb
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        for name, model in self.model_dict.items():
            print(f"\n开始训练 {name} ...")
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            self.trained_models[name] = {"model": model, "train_time": train_time}
            print(f"{name} 训练耗时：{train_time:.2f}s")
        return self.trained_models

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
        model.predict(X[:1000])
        infer_time = (time.time() - t0)/1000
        return {"accuracy": acc, "infer_ms_per_sample": infer_time*1000}