from data_loader import DataLoader
from preprocess import DataPreprocessor
from models import ModelZoo
from train_eval import Evaluator

def main():
    print("===== 干豆多分类机器学习项目启动 =====")
    # 1. 加载数据+EDA分析
    loader = DataLoader()
    df_tr, df_val, df_te = loader.load_all_data()
    loader.eda_analysis()

    # 2. 数据清洗+特征工程
    pre = DataPreprocessor()
    df_tr_clean = pre.clean_dirty_data(df_tr)
    df_val_clean = pre.clean_dirty_data(df_val)
    df_te_clean = pre.clean_dirty_data(df_te)
    train_set, val_set, test_set = pre.feature_engineering(df_tr_clean, df_val_clean, df_te_clean)

    # 3. 训练全部模型（返回训练历史用于绘制 loss 曲线）
    zoo = ModelZoo()
    trained_models, training_histories = zoo.train_all(train_set[0], train_set[1])

    # 4. 完整评估、绘图、输出对比表
    evaluator = Evaluator()
    evaluator.run_full_eval(trained_models, train_set, val_set, test_set, pre, training_histories)
    print("\n===== 项目全流程执行完毕！查看output文件夹获取图表与结果 =====")

if __name__ == "__main__":
    main()