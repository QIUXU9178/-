#!/usr/bin/env python3
"""
命令行统一入口：提供数据准备、训练、评估与生成报告的子命令。
在项目根目录运行，例如：
  python run_cli.py all
  python run_cli.py train
  python run_cli.py eval
  python run_cli.py report
"""
import argparse
import sys
from pathlib import Path

# 确保可以导入 src 下模块
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'src'))

from run import main as run_main
from data_loader import DataLoader
from preprocess import DataPreprocessor
from models import ModelZoo
from train_eval import Evaluator

def cmd_all():
    run_main()

def cmd_train():
    print('执行训练阶段...')
    loader = DataLoader()
    df_tr, df_val, df_te = loader.load_all_data()
    pre = DataPreprocessor()
    df_tr_clean = pre.clean_dirty_data(df_tr)
    df_val_clean = pre.clean_dirty_data(df_val)
    df_te_clean = pre.clean_dirty_data(df_te)
    train_set, val_set, test_set = pre.feature_engineering(df_tr_clean, df_val_clean, df_te_clean)
    zoo = ModelZoo()
    trained_models, training_histories = zoo.train_all(train_set[0], train_set[1])
    print('训练完成。可运行 `python run_cli.py eval` 进行评估')

def cmd_eval():
    print('执行评估阶段...')
    loader = DataLoader()
    df_tr, df_val, df_te = loader.load_all_data()
    pre = DataPreprocessor()
    train_set, val_set, test_set = pre.feature_engineering(pre.clean_dirty_data(df_tr), pre.clean_dirty_data(df_val), pre.clean_dirty_data(df_te))
    zoo = ModelZoo()
    trained_models, training_histories = zoo.train_all(train_set[0], train_set[1])
    evaluator = Evaluator()
    evaluator.run_full_eval(trained_models, train_set, val_set, test_set, pre, training_histories)

def cmd_report():
    print('生成论文(docx)和结果表格...')
    try:
        import md_to_docx
        md_to_docx.main()
    except Exception as e:
        print('生成 docx 时出错：', e)

def main():
    parser = argparse.ArgumentParser(description='干豆多分类项目统一命令行')
    parser.add_argument('cmd', nargs='?', default='all', help='子命令: all|train|eval|report')
    args = parser.parse_args()
    if args.cmd == 'all':
        cmd_all()
    elif args.cmd == 'train':
        cmd_train()
    elif args.cmd == 'eval':
        cmd_eval()
    elif args.cmd == 'report':
        cmd_report()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
