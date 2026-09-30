import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# ==================== 参数设置 ====================
TRADING_DAYS = 252
gammas = [1, 10]

# 基准模型映射（对比模型属于哪个 HAR 家族）
benchmark_mapping = {
    'HAR_预测RV': 'HAR_预测RV',
    'HAR-avgplt_预测RV': 'HAR-avgplt_预测RV',
    'HAR-weightplt_预测RV': 'HAR-weightplt_预测RV',
    'HAR-weather_预测RV': 'HAR-weather_预测RV',
    'HAR-clmfactor_预测RV': 'HAR-clmfactor_预测RV',
    'HAR-weafactor_预测RV': 'HAR-weafactor_预测RV',
    'HAR-all_预测RV': 'HAR-all_预测RV',

    'PLS-HAR-avgplt_预测RV': 'HAR-avgplt_预测RV',
    'PLS-HAR-weightplt_预测RV': 'HAR-weightplt_预测RV',
    'PLS-HAR-weather_预测RV': 'HAR-weather_预测RV',
    'PLS-HAR-clmfactor_预测RV': 'HAR-clmfactor_预测RV',
    'PLS-HAR-weafactor_预测RV': 'HAR-weafactor_预测RV',
    'PLS-HAR-all_预测RV': 'HAR-all_预测RV',

    'LASSO-HAR-avgplt_预测RV': 'HAR-avgplt_预测RV',
    'LASSO-HAR-weightplt_预测RV': 'HAR-weightplt_预测RV',
    'LASSO-HAR-weather_预测RV': 'HAR-weather_预测RV',
    'LASSO-HAR-clmfactor_预测RV': 'HAR-clmfactor_预测RV',
    'LASSO-HAR-weafactor_预测RV': 'HAR-weafactor_预测RV',
    'LASSO-HAR-all_预测RV': 'HAR-all_预测RV',

    'BMA-HAR-avgplt_预测RV': 'HAR-avgplt_预测RV',
    'BMA-HAR-weightplt_预测RV': 'HAR-weightplt_预测RV',
    'BMA-HAR-weather_预测RV': 'HAR-weather_预测RV',
    'BMA-HAR-clmfactor_预测RV': 'HAR-clmfactor_预测RV',
    'BMA-HAR-weafactor_预测RV': 'HAR-weafactor_预测RV',
    'BMA-HAR-all_预测RV': 'HAR-all_预测RV',

    'Enet-HAR-avgplt_预测RV': 'HAR-avgplt_预测RV',
    'Enet-HAR-weightplt_预测RV': 'HAR-weightplt_预测RV',
    'Enet-HAR-weather_预测RV': 'HAR-weather_预测RV',
    'Enet-HAR-clmfactor_预测RV': 'HAR-clmfactor_预测RV',
    'Enet-HAR-weafactor_预测RV': 'HAR-weafactor_预测RV',
    'Enet-HAR-all_预测RV': 'HAR-all_预测RV',
}

# ==================== 标准二次效用函数 ====================
def quadratic_utility(r, gamma):
    """
    标准二次效用函数
    r : 组合总收益率（小数）
    gamma : 风险厌恶系数
    """
    coef = gamma / (2 * (1 + gamma) ** 2)
    return (1 + r) - coef * (1 + r) ** 2

# ==================== 求解 Delta ====================
def objective(delta, returns_l, returns_k, gamma):
    """
    delta : 待求解的日度收益差（小数）
    方程：sum U(r_l) = sum U(r_k - delta)
    """
    return np.sum(quadratic_utility(returns_l, gamma)) - np.sum(quadratic_utility(returns_k - delta, gamma))

# ==================== 主程序 ====================
file_path = r"C:\Users\35632\Desktop\经济效益检验.xlsx"  # 请修改为实际路径
sheet_names = ['1', '5', '22']   # 三个步长对应的sheet

all_delta_results = []

for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()   # 去除列名首尾空格

    # 计算日度数据
    df['r_daily'] = df['return'] - 1                    # 日总收益率
    df['rf_daily'] = df['rf'] / 100 / TRADING_DAYS     # 日无风险收益率
    df['r_ex'] = df['r_daily'] - df['rf_daily']        # 日超额收益率

    # 预测收益率 μ_t : 滚动窗口 252 天的均值，shift(1) 避免前瞻
    df['mu_ex'] = df['r_ex'].rolling(window=TRADING_DAYS, min_periods=1).mean().shift(1)

    model_cols = [c for c in df.columns if '预测RV' in c]

    for model_l in model_cols:
        if model_l not in benchmark_mapping:
            continue
        model_k = benchmark_mapping[model_l]
        if model_k == model_l:
            continue   # 跳过自身对比

        # 有效数据：mu_ex 和两个模型预测方差均非空
        valid = df.dropna(subset=['mu_ex', model_l, model_k, 'r_daily', 'rf_daily']).copy()
        if len(valid) < 20:
            continue

        for gamma in gammas:
            # 基准模型 k 的组合收益率
            valid['w_k'] = valid['mu_ex'] / (gamma * valid[model_k])
            valid['rp_k'] = valid['w_k'] * valid['r_daily'] + (1 - valid['w_k']) * valid['rf_daily']

            # 对比模型 l 的组合收益率
            valid['w_l'] = valid['mu_ex'] / (gamma * valid[model_l])
            valid['rp_l'] = valid['w_l'] * valid['r_daily'] + (1 - valid['w_l']) * valid['rf_daily']

            r_k = valid['rp_k'].values
            r_l = valid['rp_l'].values

            # 求解 Δ（日度值）
            delta0 = 0.0
            try:
                delta_sol = fsolve(objective, delta0, args=(r_l, r_k, gamma), xtol=1e-8)
                delta_daily = delta_sol[0]
            except:
                delta_daily = np.nan

            # 年化并转为百分比
            delta_annual_pct = delta_daily * TRADING_DAYS * 100

            # 存储结果
            all_delta_results.append({
                'Sheet': sheet_name,
                'Benchmark': model_k,
                'Model': model_l,
                'Gamma': gamma,
                'Delta (%)': delta_annual_pct
            })

# ==================== 输出结果 ====================
results_df = pd.DataFrame(all_delta_results)
# 保存长格式（每个模型在每个 gamma 下一行）
output_path = r"C:\Users\35632\Desktop\经济价值_Delta_标准效用.xlsx"
results_df.to_excel(output_path, index=False)

print(f"计算完成，结果已保存至：{output_path}")
print("\n前10行预览：")
print(results_df.head(10))

# ========== 可选：转换为宽表（每个模型一行，两列 gamma） ==========
# 如果需要宽表，取消下面注释
# wide_df = results_df.pivot_table(index=['Sheet', 'Benchmark', 'Model'], columns='Gamma', values='Delta (%)')
# wide_df.columns = [f'Delta_gamma{col} (%)' for col in wide_df.columns]
# wide_df.reset_index().to_excel(r"C:\Users\35632\Desktop\经济价值_Delta_标准效用_宽表.xlsx", index=False)
# print("宽表也已保存。")