import pandas as pd
import numpy as np

# 文件路径
file_path = r"C:\Users\35632\Desktop\经济效益检验.xlsx"

# 读取所有sheet
xls = pd.ExcelFile(file_path)
sheet_names = ['1', '5', '22']

# 假设每年交易日数
TRADING_DAYS = 252

# 风险厌恶系数
gammas = [1, 10]

# 用于汇总结果的列表
all_results = []

for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    # 列名: 日期, close, return, rf, 真实RV, ... 预测模型列
    # return 是 1 + r_daily
    df['r_daily'] = df['return'] - 1  # 实际日收益率 r_t
    df['rf_daily'] = df['rf'] / 100 / TRADING_DAYS  # 日无风险收益率
    df['r_ex'] = df['r_daily'] - df['rf_daily']  # 日超额收益率

    # 模型列从 'HAR_预测RV' 开始到末尾
    model_columns = [col for col in df.columns if '_预测RV' in col]

    # 基准模型是 'HAR_预测RV'
    benchmark_model = 'HAR_预测RV'

    # 为每个模型计算
    for model in model_columns:
        predicted_rv = df[model]  # 预测RV (假设为variance)

        # 使用 expanding mean for mu_t (从第2行开始计算，第一个无mu)
        df['mu'] = df['r_ex'].expanding().mean().shift(1)  # t时刻用t-1前的平均

        # 跳过 mu 为 NaN 的行
        valid_df = df.dropna(subset=['mu', model])

        for gamma in gammas:
            # 计算权重 w_t = mu_t / (gamma * predicted_var_t)
            valid_df['w'] = valid_df['mu'] / (gamma * valid_df[model])

            # 组合超额收益 p_ex_t = w_t * r_ex_t
            valid_df['p_ex'] = valid_df['w'] * valid_df['r_ex']

            # 年化平均超额收益
            ann_ex_return = valid_df['p_ex'].mean() * TRADING_DAYS

            # 年化夏普比率
            if valid_df['p_ex'].std() > 0:
                sr = valid_df['p_ex'].mean() / valid_df['p_ex'].std() * np.sqrt(TRADING_DAYS)
            else:
                sr = np.nan

            # 保存结果
            all_results.append({
                'Sheet': sheet,
                'Model': model,
                'Gamma': gamma,
                'Ann_Ex_Return (%)': ann_ex_return * 100,
                'Sharpe_Ratio': sr
            })

# 转为DataFrame
results_df = pd.DataFrame(all_results)

# 排序
results_df = results_df.sort_values(['Sheet', 'Gamma', 'Model'])

# 保存到Excel
output_path = r"C:\Users\35632\Desktop\Economic_Evaluation_Results.xlsx"
results_df.to_excel(output_path, index=False)

print(f"结果已保存到 {output_path}")

# 打印预览
print(results_df)