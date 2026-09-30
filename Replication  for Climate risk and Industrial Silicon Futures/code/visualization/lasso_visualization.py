# -*- coding: utf-8 -*-
"""
LASSO-HAR模型 - 完整版
添加交叉验证误差曲线 - 优化纵坐标尺度
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import openpyxl
from scipy import stats
import os
import platform

# 设置中文字体和负号显示 - 彻底修复
# 根据操作系统选择合适的字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    # 对于负号，使用标准ASCII负号
    plt.rcParams['axes.unicode_minus'] = False
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 或者使用最简单粗暴的方法：全局设置使用标准ASCII字符
mpl.rcParams['mathtext.default'] = 'regular'  # 数学文本使用普通字体
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.bf'] = 'sans:bold'

# 忽略所有字体警告
warnings.filterwarnings('ignore', message='Font .* does not have a glyph for')
warnings.filterwarnings('ignore')

print(f"字体设置: {plt.rcParams['font.sans-serif']}")
print(f"负号设置: unicode_minus = {plt.rcParams['axes.unicode_minus']}")

# ----------------------
# 1. 全局配置
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\LASSO-HAR模型_完整版.xlsx"
fig_save_path = r"C:\Users\35632\Desktop\LASSO-HAR可视化结果_完整版"

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
horizon_lag_required = {
    1: ["daily"],
    5: ["daily", "weekly"],
    22: ["daily", "weekly", "monthly"]
}


# ----------------------
# 工具函数
# ----------------------
def calculate_loss(y_true, y_pred):
    valid_idx = (y_pred > 0) & (y_true > 0)
    if not np.any(valid_idx):
        return 0.0, 0.0, 0.0, 0.0
    y_true_v = y_true[valid_idx]
    y_pred_v = y_pred[valid_idx]
    mse = mean_squared_error(y_true_v, y_pred_v)
    mae = mean_absolute_error(y_true_v, y_pred_v)
    qlike = np.mean(np.log(y_pred_v) + y_true_v / y_pred_v)
    r2log = np.mean((np.log(y_true_v) - np.log(y_pred_v)) ** 2)
    return mse, mae, qlike, r2log


def clean_data(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if col == 'date':
            continue
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.replace('%', '').replace('nan', np.nan)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        fill_val = df_clean[col].median() if df_clean[col].notna().sum() > 0 else 0
        df_clean[col] = df_clean[col].fillna(fill_val)
    return df_clean


def pca_analysis_enhanced(data, cols, n_components, name):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_scaled)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'{name}_PC{i + 1}' for i in range(n_components)],
        index=cols
    )
    var_contribution = np.abs(loadings).sum(axis=1)
    var_contribution = var_contribution / var_contribution.sum() * 100
    loadings_df['总贡献度(%)'] = var_contribution.round(2)

    exp_var = pca.explained_variance_ratio_ * 100
    exp_var_df = pd.DataFrame({
        '主成分': [f'{name}_PC{i + 1}' for i in range(n_components)],
        '方差贡献率(%)': exp_var.round(2),
        '累计贡献率(%)': np.cumsum(exp_var).round(2)
    })

    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df.iloc[:, :-1], cmap='RdBu_r', annot=True, fmt='.2f')
    plt.title(f'{name} 主成分载荷矩阵')
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{name}_PCA载荷矩阵.png', dpi=300)
    plt.close()

    return data_pca, loadings_df, exp_var_df, scaler


def generate_lags(data, n_samples):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    lags = {}
    lags["daily"] = np.roll(data, 1, axis=0)

    weekly = np.zeros_like(data)
    for i in range(5, n_samples):
        weekly[i] = np.mean(data[i - 5:i], axis=0)
    lags["weekly"] = weekly

    monthly = np.zeros_like(data)
    for i in range(22, n_samples):
        monthly[i] = np.mean(data[i - 22:i], axis=0)
    lags["monthly"] = monthly

    for lag_type in lags:
        lags[lag_type][:22] = np.nanmean(lags[lag_type][22:], axis=0)

    return lags


def build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, horizon):
    required_lags = horizon_lag_required[horizon]

    rv_matrix = []
    rv_var_names = []
    for lag_type in required_lags:
        rv_col = rv_lags[lag_type]
        rv_matrix.append(rv_col)
        rv_var_names.append(f'RV_{lag_type}')
    rv_matrix = np.hstack(rv_matrix) if len(rv_matrix) > 1 else rv_matrix[0]

    if exog_lags is None:
        return rv_matrix, rv_var_names

    exog_matrix = []
    exog_var_names_full = []
    for lag_type in required_lags:
        exog_col = exog_lags[lag_type]
        exog_matrix.append(exog_col)
        exog_var_names_lag = [f'{var}_{lag_type}' for var in exog_var_names]
        exog_var_names_full.extend(exog_var_names_lag)
    exog_matrix = np.hstack(exog_matrix) if len(exog_matrix) > 1 else exog_matrix[0]

    X_matrix = np.hstack([rv_matrix, exog_matrix])
    full_var_names = rv_var_names + exog_var_names_full

    return X_matrix, full_var_names


# ----------------------
# 绘制LASSO交叉验证误差曲线 - 优化纵坐标尺度
# ----------------------
def plot_lasso_cv_curve(X, y, var_names, model_name, horizon_name, best_lambda, fig_save_path):
    """绘制LASSO交叉验证误差随λ变化的曲线 - 优化纵坐标尺度"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

    # 使用更多alpha值以获得平滑曲线
    tscv = TimeSeriesSplit(n_splits=5)
    lasso_cv = LassoCV(
        cv=tscv,
        n_alphas=200,
        random_state=42,
        max_iter=10000,
        tol=1e-3
    )
    lasso_cv.fit(X_scaled, y_scaled)

    # 提取MSE路径
    alphas = lasso_cv.alphas_
    mse_path = lasso_cv.mse_path_
    mse_mean = np.mean(mse_path, axis=1)
    mse_std = np.std(mse_path, axis=1)

    # 计算MSE的相对变化
    mse_min = np.min(mse_mean)
    mse_max = np.max(mse_mean)
    mse_range = mse_max - mse_min

    # 设置纵坐标范围：最小值的0.9倍到最大值的1.1倍，但至少保证有10%的变化空间
    y_min = max(0, mse_min * 0.9)
    y_max = mse_max * 1.1

    # 如果变化太小（小于1e-6），则使用固定范围
    if mse_range < 1e-6:
        y_min = mse_min * 0.5
        y_max = mse_min * 1.5

    # 绘图
    plt.figure(figsize=(12, 8))

    # 绘制MSE均值曲线
    plt.plot(alphas, mse_mean, 'b-', label='平均MSE', linewidth=2)

    # 绘制标准差带
    plt.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2, color='blue')

    # 标记最优λ
    plt.axvline(x=best_lambda, color='red', linestyle='--', linewidth=2,
                label=f'最优λ = {best_lambda:.6f}')

    # 标记最小MSE点
    min_idx = np.argmin(mse_mean)
    plt.plot(alphas[min_idx], mse_mean[min_idx], 'ro', markersize=10,
             label=f'最小MSE = {mse_mean[min_idx]:.8f}')

    # 设置纵坐标范围
    plt.ylim(y_min, y_max)

    # 添加水平参考线
    plt.axhline(y=mse_min, color='gray', linestyle=':', alpha=0.5, label=f'最小MSE线')

    plt.xscale('log')
    plt.xlabel('λ (对数刻度)', fontsize=12)
    plt.ylabel('交叉验证MSE', fontsize=12)
    plt.title(f'{model_name} - {horizon_name} LASSO交叉验证误差曲线', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{model_name}_{horizon_name}_CV曲线.png', dpi=300, bbox_inches='tight')
    plt.close()

    return alphas, mse_mean, mse_std


def lasso_variable_selection_enhanced(X, y, horizon, var_names):
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    tscv = TimeSeriesSplit(n_splits=5)
    lasso_cv = LassoCV(
        cv=tscv,
        n_alphas=100,
        random_state=42,
        max_iter=10000,
        tol=1e-3
    )
    lasso_cv.fit(X_scaled, y_scaled)

    best_lambda = lasso_cv.alpha_
    non_zero_idx = np.where(lasso_cv.coef_ != 0)[0]

    required_lags = horizon_lag_required[horizon]
    n_required_har = len(required_lags)
    if len(non_zero_idx) < n_required_har:
        top_idx = np.argsort(np.abs(lasso_cv.coef_))[::-1][:n_required_har]
        selected_idx = top_idx
    else:
        selected_idx = non_zero_idx

    selected_idx = selected_idx[np.argsort(np.abs(lasso_cv.coef_[selected_idx]))[::-1]]
    X_selected = X[:, selected_idx]
    selected_var_names = [var_names[i] for i in selected_idx]

    return X_selected, selected_idx, selected_var_names, best_lambda, scaler_X, scaler_y, lasso_cv.coef_


# ----------------------
# 2. 数据加载与预处理
# ----------------------
print("正在加载数据...")
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

rv = df_clean['rv'].values
rv_log = np.log1p(rv)
n_samples = len(rv)
rv_lags = generate_lags(rv_log, n_samples)

# 提取外生因子
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_lags(X_avgplt, n_samples)

weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_lags(X_weightplt, n_samples)

weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_lags(X_weather, n_samples)

clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
X_clmfactor, clm_loadings, clm_exp_var, clm_scaler = pca_analysis_enhanced(
    X_clmfactor_raw, clmfactor_cols, 3, 'CLMFactor'
)
exog_lags_clmfactor = generate_lags(X_clmfactor, n_samples)
clmfactor_pc_names = [f'CLMFactor_PC{i + 1}' for i in range(3)]

weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
X_weafactor, wea_loadings, wea_exp_var, wea_scaler = pca_analysis_enhanced(
    X_weafactor_raw, weafactor_cols, 3, 'WEAFactor'
)
exog_lags_weafactor = generate_lags(X_weafactor, n_samples)
weafactor_pc_names = [f'WEAFactor_PC{i + 1}' for i in range(3)]

X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_lags(X_all, n_samples)
all_exog_names = avgplt_cols + weightplt_cols + weather_cols + clmfactor_pc_names + weafactor_pc_names

models_dict = {
    "LASSO-HAR-avgplt": (exog_lags_avgplt, avgplt_cols),
    "LASSO-HAR-weightplt": (exog_lags_weightplt, weightplt_cols),
    "LASSO-HAR-weather": (exog_lags_weather, weather_cols),
    "LASSO-HAR-clmfactor": (exog_lags_clmfactor, clmfactor_pc_names),
    "LASSO-HAR-weafactor": (exog_lags_weafactor, weafactor_pc_names),
    "LASSO-HAR-all": (exog_lags_all, all_exog_names)
}

valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测 + 交叉验证曲线
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons}
                    for model in models_dict.keys()}
lasso_detailed_info = {model: {h: [] for h in forecast_horizons} for model in models_dict.keys()}
cv_curve_info = {model: {h: None for h in forecast_horizons} for model in models_dict.keys()}

for h in forecast_horizons:
    horizon_name = horizon_names[forecast_horizons.index(h)]
    print(f"\n=== 开始{h}步超前预测（{horizon_name}）===")

    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])

    total_windows = len(y_target) - window_size
    for i in range(window_size, len(y_target)):
        if i % 50 == 0:
            print(f"  进度: {i - window_size + 1}/{total_windows}")

        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        test_idx_raw = valid_start + i
        pred_date = dates_valid[i + h - 1]
        y_train = np.nan_to_num(y_target[i - window_size: i], nan=np.nanmean(y_target))
        true_rv = np.expm1(y_target[i])

        for model_name, (exog_lags, exog_var_names) in models_dict.items():
            X_raw, var_names = build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, h)

            X_train = X_raw[train_idx_raw]
            X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
            X_test = X_raw[test_idx_raw].reshape(1, -1)
            X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))

            # LASSO变量选择
            X_lasso, selected_idx, selected_var_names, best_lambda, scaler_X, scaler_y, all_coef = \
                lasso_variable_selection_enhanced(X_train, y_train, h, var_names)

            # 在第一个窗口绘制交叉验证曲线
            if len(lasso_detailed_info[model_name][h]) == 0:
                print(f"  绘制{model_name}的交叉验证曲线...")
                alphas, mse_mean, mse_std = plot_lasso_cv_curve(
                    X_train, y_train, var_names, model_name, horizon_name, best_lambda, fig_save_path
                )
                cv_curve_info[model_name][h] = {
                    'alphas': alphas,
                    'mse_mean': mse_mean,
                    'mse_std': mse_std,
                    'best_lambda': best_lambda
                }

            # 记录LASSO信息
            lasso_detailed_info[model_name][h].append({
                'date': pred_date,
                'selected_idx': selected_idx,
                'selected_var_names': selected_var_names,
                'best_lambda': best_lambda,
                'all_coef': all_coef,
                'original_dim': X_train.shape[1],
                'selected_dim': len(selected_idx)
            })

            # 标准化+预测
            X_train_scaled = scaler_X.fit_transform(X_lasso)
            X_test_lasso = X_test[:, selected_idx]
            X_test_scaled = scaler_X.transform(X_test_lasso)

            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred_log = model.predict(X_test_scaled)[0]
            y_pred_rv = np.expm1(y_pred_log)

            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

# ----------------------
# 4. 结果整理与可视化
# ----------------------
# 4.1 统计指标
stats_results = []
for model_name in models_dict.keys():
    for h in forecast_horizons:
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        first_lasso = lasso_detailed_info[model_name][h][0] if lasso_detailed_info[model_name][h] else {}

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "有效样本数": valid_len,
            "原始变量数": first_lasso.get("original_dim", 0),
            "保留变量数": first_lasso.get("selected_dim", 0),
            "最优λ": round(first_lasso.get("best_lambda", 0), 6),
            "MSE": round(mse, 8),
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# 4.2 变量选择汇总
var_selection_summary = []
for model_name in models_dict.keys():
    for h in forecast_horizons:
        horizon_name = horizon_names[forecast_horizons.index(h)]
        selected_counts = {}
        total_records = len(lasso_detailed_info[model_name][h])

        for record in lasso_detailed_info[model_name][h]:
            for var in record['selected_var_names']:
                selected_counts[var] = selected_counts.get(var, 0) + 1

        for var, count in selected_counts.items():
            freq = count / total_records * 100 if total_records > 0 else 0
            var_selection_summary.append({
                "模型名称": model_name,
                "预测步数": horizon_name,
                "变量名称": var,
                "选中次数": count,
                "总预测次数": total_records,
                "选中频率(%)": round(freq, 2)
            })

var_selection_df = pd.DataFrame(var_selection_summary)

# 4.3 可视化变量选中频率
for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]
    plt.figure(figsize=(14, 10))
    subset = var_selection_df[var_selection_df['预测步数'] == horizon_name]

    if len(subset) > 0:
        # 选择每个模型前5个最重要变量
        top_vars = []
        for model_name in models_dict.keys():
            model_data = subset[subset['模型名称'] == model_name]
            if len(model_data) > 0:
                top_vars.extend(model_data.nlargest(min(5, len(model_data)), '选中频率(%)')['变量名称'].tolist())

        top_vars_unique = list(dict.fromkeys(top_vars))[:15]
        plot_data = subset[subset['变量名称'].isin(top_vars_unique)]

        if len(plot_data) > 0:
            plt.figure(figsize=(14, 10))
            sns.barplot(x='选中频率(%)', y='变量名称', hue='模型名称', data=plot_data)
            plt.title(f'{horizon_name} - 变量选中频率', fontsize=14)
            plt.xlabel('选中频率(%)')
            plt.tight_layout()
            plt.savefig(f'{fig_save_path}/{horizon_name}_变量选中频率.png', dpi=300, bbox_inches='tight')
            plt.close()

# 4.4 可视化预测结果
for model_name in models_dict.keys():
    plt.figure(figsize=(15, 10))
    plot_count = 0
    for h_idx, h in enumerate(forecast_horizons):
        dates = predictions_dict[model_name][h]['dates']
        pred = predictions_dict[model_name][h]['pred']
        true = predictions_dict[model_name][h]['true']

        if len(dates) > 0:
            plot_count += 1
            plt.subplot(3, 1, plot_count)
            plt.plot(dates, true, label='真实值', color='blue', linewidth=1)
            plt.plot(dates, pred, label='预测值', color='red', linewidth=1, alpha=0.7)
            plt.title(f'{model_name} - {horizon_names[h_idx]}', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

    if plot_count > 0:
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}/{model_name}_预测结果.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4.5 绘制各模型的交叉验证曲线汇总图
plt.figure(figsize=(15, 12))
plot_idx = 1
for model_name in models_dict.keys():
    for h_idx, h in enumerate(forecast_horizons):
        if cv_curve_info[model_name][h] is not None and plot_idx <= 6:
            plt.subplot(3, 2, plot_idx)
            info = cv_curve_info[model_name][h]
            alphas = info['alphas']
            mse_mean = info['mse_mean']
            mse_std = info['mse_std']
            best_lambda = info['best_lambda']

            # 计算合适的纵坐标范围
            mse_min = np.min(mse_mean)
            mse_max = np.max(mse_mean)
            y_min = max(0, mse_min * 0.9)
            y_max = mse_max * 1.1

            plt.plot(alphas, mse_mean, 'b-', linewidth=2)
            plt.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2)
            plt.axvline(x=best_lambda, color='red', linestyle='--', linewidth=2)
            plt.ylim(y_min, y_max)
            plt.xscale('log')
            plt.xlabel('λ')
            plt.ylabel('MSE')
            plt.title(f'{model_name}\n{horizon_names[h_idx]}', fontsize=10)
            plt.grid(True, alpha=0.3)
            plot_idx += 1

plt.tight_layout()
plt.savefig(f'{fig_save_path}/所有模型CV曲线汇总.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# 5. 结果保存
# ----------------------
print(f"\n正在保存结果到: {output_path}")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 统计指标
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)

    # LASSO变量选择
    var_selection_df.to_excel(writer, sheet_name="LASSO变量选择", index=False)

    # PCA结果
    clm_loadings.to_excel(writer, sheet_name="CLMFactor_PCA载荷", index=True)
    clm_exp_var.to_excel(writer, sheet_name="CLMFactor_PCA贡献率", index=False)
    wea_loadings.to_excel(writer, sheet_name="WEAFactor_PCA载荷", index=True)
    wea_exp_var.to_excel(writer, sheet_name="WEAFactor_PCA贡献率", index=False)

    # 预测值
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        # 取第一个模型作为基准
        first_model = list(models_dict.keys())[0]
        if len(predictions_dict[first_model][h]['dates']) > 0:
            step_data = pd.DataFrame({"日期": predictions_dict[first_model][h]['dates']})
            for model_name in models_dict.keys():
                step_data[f'{model_name}_预测RV'] = [round(val, 6) for val in predictions_dict[model_name][h]['pred']]
                step_data[f'{model_name}_真实RV'] = [round(val, 6) for val in predictions_dict[model_name][h]['true']]
            step_data.to_excel(writer, sheet_name=f"{horizon_name}预测值", index=False)

# ----------------------
# 6. 输出汇总
# ----------------------
print(f"\n✅ 所有结果已保存至：{output_path}")
print(f"📊 可视化结果已保存至：{fig_save_path}")
print("\n📊 统计指标汇总：")
print(stats_df[["模型名称", "预测步数", "保留变量数", "最优λ", "MSE", "QLIKE"]].head(12))

print("\n🏆 各预测步数最佳模型（按MSE）：")
for h_idx, h in enumerate(forecast_horizons):
    h_stats = stats_df[stats_df['预测步数'] == horizon_names[h_idx]]
    if not h_stats.empty:
        best = h_stats.loc[h_stats['MSE'].idxmin()]
        print(f"  {horizon_names[h_idx]}: {best['模型名称']} (MSE={best['MSE']:.8f}, 保留变量={best['保留变量数']})")

print("\n✅ LASSO-HAR模型分析完成！")