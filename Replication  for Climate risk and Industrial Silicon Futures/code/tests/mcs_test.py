import numpy as np
import pandas as pd
from model_confidence_set import ModelConfidenceSet

# 文件路径
file_path = r"C:\Users\35632\Desktop\MSC稳健性检验.xlsx"

# 读取Excel文件的所有sheet
xls = pd.ExcelFile(file_path)
sheet_names = ['1', '5', '22']  # 对应提供的sheet name

# 定义损失函数
def compute_mse(actual, pred):
    return np.mean((actual - pred) ** 2)

def compute_mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def compute_qlike(actual, pred):
    # 避免log(0)或除0，添加小epsilon
    epsilon = 1e-10
    pred = np.maximum(pred, epsilon)
    return np.mean(np.log(pred) + actual / pred)

def compute_r2log(actual, pred):
    # 避免log(0)，添加小epsilon
    epsilon = 1e-10
    actual = np.maximum(actual, epsilon)
    pred = np.maximum(pred, epsilon)
    return np.mean((np.log(actual) - np.log(pred)) ** 2)

# 损失函数字典
loss_functions = {
    'MSE': compute_mse,
    'MAE': compute_mae,
    'QLIKE': compute_qlike,
    'R2LOG': compute_r2log
}

# 用于收集所有结果
all_results = []

# 循环处理每个sheet
for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    # 第一列日期，第二列真实RV，从第三列开始是模型预测
    actual = df.iloc[:, 1].values  # 真实RV (numpy array)
    model_columns = df.columns[2:]  # 模型列名
    num_models = len(model_columns)
    num_obs = len(actual)

    print(f"\nProcessing Sheet: {sheet}")
    print(f"Number of observations: {num_obs}")
    print(f"Number of models: {num_models}")

    # 为每个损失函数计算losses矩阵 (num_obs x num_models)
    for loss_name, loss_func in loss_functions.items():
        losses = np.zeros((num_obs, num_models))

        for i, model_col in enumerate(model_columns):
            pred = df[model_col].values  # 模型预测 (numpy array)
            # 计算每个观测的损失
            for j in range(num_obs):
                losses[j, i] = loss_func(actual[j], pred[j])

        print(f"  计算 {loss_name} ...", end=" ", flush=True)

        # 初始化MCS
        mcs = ModelConfidenceSet(losses, n_boot=5000, alpha=0.05, show_progress=True)

        # 计算
        mcs.compute()

        # 获取结果
        results = mcs.results(as_dataframe=True)

        # ─────────────── 关键：添加模型名称 ───────────────
        # 假设 results 的行顺序与 model_columns 一致
        results = results.reset_index()  # 把可能的索引变成列
        if 'index' in results.columns:
            results = results.rename(columns={'index': 'Model'})
        else:
            # 如果没有 index 列，直接加一列 Model
            results['Model'] = model_columns

        # 添加 sheet 和 loss 标识
        results['Sheet'] = sheet
        results['Loss'] = loss_name

        # 调整列顺序（把 Sheet, Loss, Model 放前面）
        cols = ['Sheet', 'Loss', 'Model'] + [c for c in results.columns if c not in ['Sheet', 'Loss', 'Model']]
        results = results[cols]

        all_results.append(results)

        print("完成")

        # 可选：单个文件保存
        # results.to_csv(f"MCS_{loss_name}_Sheet_{sheet}.csv", index=False)

# ─────────────── 合并所有结果到一张表 ───────────────
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)

    # 排序方便查看
    final_df = final_df.sort_values(['Sheet', 'Loss', 'Model'])

    # 保存到 Excel
    output_excel = r"C:\Users\35632\Desktop\MCS_检验wenjianx.xlsx"
    final_df.to_excel(output_excel, index=False)
    print(f"\n所有结果已保存到：{output_excel}")

    # 打印列名，方便你知道 p 值叫什么名字
    print("\n汇总表的列名：")
    print(final_df.columns.tolist())

    # ─────────────── 自动找 p 值列并统计 ───────────────
    p_col_candidates = [col for col in final_df.columns if 'p' in col.lower()]
    if p_col_candidates:
        p_col = p_col_candidates[0]  # 取第一个包含 p 的列
        print(f"\n检测到的 p 值列名：{p_col}")

        # 统计 p 值 > 0.25 的模型数量
        stats = final_df.groupby(['Sheet', 'Loss'])[p_col].apply(lambda x: (x > 0.25).sum()).reset_index(name='p > 0.25 的模型数')
        print("\n简单统计（p 值 > 0.25 的模型数量）：")
        print(stats)

        # 额外多看几个阈值
        print("\n多阈值统计：")
        multi_stats = final_df.groupby(['Sheet', 'Loss'])[p_col].agg([
            lambda x: (x > 0.25).sum(),
            lambda x: (x > 0.10).sum(),
            lambda x: (x > 0.05).sum()
        ]).rename(columns={
            '<lambda_0>': 'p > 0.25',
            '<lambda_1>': 'p > 0.10',
            '<lambda_2>': 'p > 0.05'
        })
        print(multi_stats)
    else:
        print("\n警告：没有找到任何包含 'p' 的列，请检查 MCS 库输出格式")
        print("前5行数据样子：")
        print(final_df.head())
else:
    print("没有生成任何结果，请检查前面是否有报错")