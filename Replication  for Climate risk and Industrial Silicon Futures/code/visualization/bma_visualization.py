# -*- coding: utf-8 -*-
"""
BMA-HAR模型 - 完整版
添加模型后验概率分布和BIC值分布可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import openpyxl
import os
from itertools import combinations
from scipy import stats
import time

# 设置中文字体（解决可视化中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置与核心工具函数
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\HAR_BMA模型结果_增强版.xlsx"
fig_save_path = r"C:\Users\35632\Desktop\HAR_BMA可视化结果_增强版"

forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
# 核心映射：预测步数→需纳入的滞后项类型（严格贴合你的逻辑）
horizon_lag_map = {
    1: ["daily"],  # 1步：仅日度（前1期）
    5: ["daily", "weekly"],  # 5步：日度+周度（前5期均值）
    22: ["daily", "weekly", "monthly"]  # 22步：日度+周度+月度（前22期均值）
}

# BMA参数
BMA_MAX_VARS = 5  # 每个模型最大变量数（避免组合爆炸）
BMA_PRIOR_VAR = 1e6  # 回归系数的先验方差（Zellner's g-prior）
BMA_MIN_PROB = 0.01  # 最小后验概率阈值
TOP_N_MODELS = 10  # 显示前N个最优模型

# 创建可视化保存目录
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)


# 统计损失函数（严格匹配论文公式）
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


# 数据清理函数
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


# 主成分分析辅助函数
def pca_analysis(data, cols, n_components, name):
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)

    # 计算载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'{name}_PC{i + 1}' for i in range(n_components)],
        index=cols
    )

    # 计算各变量对主成分的贡献度
    var_contribution = np.abs(loadings).sum(axis=1)
    var_contribution = var_contribution / var_contribution.sum() * 100
    loadings_df['总贡献度(%)'] = var_contribution.round(2)

    # 主成分贡献率
    exp_var = pca.explained_variance_ratio_ * 100
    exp_var_df = pd.DataFrame({
        '主成分': [f'{name}_PC{i + 1}' for i in range(n_components)],
        '方差贡献率(%)': exp_var.round(2),
        '累计贡献率(%)': np.cumsum(exp_var).round(2)
    })

    # 可视化载荷矩阵
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df.iloc[:, :-1], cmap='RdBu_r', annot=True, fmt='.2f', cbar_kws={'label': '载荷系数'})
    plt.title(f'{name} 主成分载荷矩阵（{n_components}个主成分）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{name}_PCA载荷矩阵.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 可视化变量贡献度
    plt.figure(figsize=(10, 6))
    top_vars = loadings_df['总贡献度(%)'].sort_values(ascending=False).head(10)
    sns.barplot(x=top_vars.values, y=top_vars.index)
    plt.title(f'{name} 前10个核心变量贡献度', fontsize=14)
    plt.xlabel('贡献度(%)')
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{name}_核心变量贡献度.png', dpi=300, bbox_inches='tight')
    plt.close()

    return data_pca, loadings_df, exp_var_df


# ----------------------
# 核心修复1：生成三层滞后项（日度/周度/月度）
# ----------------------
def generate_all_lags(data, n_samples):
    """生成数据的三层滞后项：daily（前1期）、weekly（前5期均值）、monthly（前22期均值）"""
    n_features = data.shape[1] if len(data.shape) > 1 else 1
    if n_features == 1:
        data = data.reshape(-1, 1)

    # 1. 日度滞后（前1期，t-1）
    daily = np.roll(data, 1, axis=0)
    # 2. 周度滞后（前5期均值，t-1至t-5）
    weekly = np.zeros_like(data)
    for i in range(5, n_samples):
        weekly[i] = np.mean(data[i - 5:i], axis=0)
    # 3. 月度滞后（前22期均值，t-1至t-22）
    monthly = np.zeros_like(data)
    for i in range(22, n_samples):
        monthly[i] = np.mean(data[i - 22:i], axis=0)

    # 填充前22期缺失值
    daily[:22] = np.nanmean(daily[22:], axis=0)
    weekly[:22] = np.nanmean(weekly[22:], axis=0)
    monthly[:22] = np.nanmean(monthly[22:], axis=0)

    return {"daily": daily, "weekly": weekly, "monthly": monthly}


# ----------------------
# 核心修复2：动态构建自变量矩阵+变量名（严格匹配步数）
# ----------------------
def build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, horizon, include_rv=True):
    """
    按预测步数构建自变量矩阵+变量名：
    - 1步：仅daily（前1期）
    - 5步：daily+weekly（前1期+前5天均值）
    - 22步：daily+weekly+monthly（前1期+前5天+前22天均值）
    """
    required_lags = horizon_lag_map[horizon]

    all_matrices = []
    all_var_names = []

    # 1. 添加RV滞后项（如果需要）
    if include_rv:
        for lag_type in required_lags:
            rv_col = rv_lags[lag_type].reshape(-1, 1)
            all_matrices.append(rv_col)
            all_var_names.append(f'RV_{lag_type}')

    # 2. 添加外生变量滞后项（如果有）
    if exog_lags is not None:
        for lag_type in required_lags:
            exog_col = exog_lags[lag_type]
            all_matrices.append(exog_col)
            # 为每个外生变量添加滞后类型后缀
            exog_var_names_lag = [f'{var}_{lag_type}' for var in exog_var_names]
            all_var_names.extend(exog_var_names_lag)

    # 3. 合并所有变量
    if len(all_matrices) == 0:
        return None, []

    X_matrix = np.hstack(all_matrices)

    # 安全校验：维度一致
    assert X_matrix.shape[1] == len(all_var_names), \
        f"矩阵列数({X_matrix.shape[1]})与变量名数量({len(all_var_names)})不匹配！"

    return X_matrix, all_var_names


# ----------------------
# BMA核心函数：计算模型后验概率（增强版，返回BIC和模型信息）
# ----------------------
def calculate_bma_weights_enhanced(X_train, y_train, var_names, max_vars=BMA_MAX_VARS):
    """
    使用BMA计算各变量的后验包含概率
    基于BIC近似后验模型概率
    增强版：返回BIC值、模型信息等用于可视化
    """
    n_samples = len(y_train)
    n_vars = X_train.shape[1]

    if n_vars == 0:
        return np.array([]), {}, [], [], []

    # 标准化数据
    X_scaled = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    y_scaled = (y_train - np.mean(y_train))

    # 限制变量组合数量，避免组合爆炸
    max_combinations = 5000  # 最大尝试的模型数量

    # 生成候选模型（从简单到复杂）
    candidate_models = []

    # 1. 空模型
    candidate_models.append([])

    # 2. 单变量模型
    for i in range(min(n_vars, max_vars)):
        candidate_models.append([i])

    # 3. 双变量模型
    if n_vars >= 2:
        combs = list(combinations(range(n_vars), 2))
        candidate_models.extend([list(comb) for comb in combs[:max_combinations // 4]])

    # 4. 三变量模型
    if n_vars >= 3:
        combs = list(combinations(range(n_vars), 3))
        candidate_models.extend([list(comb) for comb in combs[:max_combinations // 4]])

    # 5. 更多变量的模型（如果需要）
    for k in range(4, min(n_vars, max_vars) + 1):
        if len(candidate_models) < max_combinations:
            combs = list(combinations(range(n_vars), k))
            candidate_models.extend([list(comb) for comb in combs[:max_combinations // 4]])

    # 计算每个模型的BIC和权重
    bics = []
    model_info = []

    for var_indices in candidate_models:
        if len(var_indices) == 0:
            # 空模型：仅截距
            y_pred = np.mean(y_scaled)
            residuals = y_scaled - y_pred
            rss = np.sum(residuals ** 2)
            k = 0  # 仅截距
            model_vars = []
            model_vars_names = []
        else:
            # 有变量的模型
            X_sub = X_scaled[:, var_indices]
            try:
                # 使用线性回归
                model = LinearRegression()
                model.fit(X_sub, y_scaled)
                y_pred = model.predict(X_sub)
                residuals = y_scaled - y_pred
                rss = np.sum(residuals ** 2)
                k = len(var_indices)
                model_vars = var_indices
                model_vars_names = [var_names[i] for i in var_indices]
            except:
                continue

        # 计算BIC
        bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
        bics.append(bic)

        # 保存模型信息
        model_info.append({
            'var_indices': model_vars,
            'var_names': model_vars_names,
            'k': k,
            'bic': bic,
            'rss': rss
        })

    # 转换为后验概率
    bics = np.array(bics)
    bics_min = np.min(bics)
    bics_normalized = bics - bics_min  # 避免数值问题
    weights = np.exp(-0.5 * bics_normalized)
    weights = weights / np.sum(weights)

    # 计算每个变量的后验包含概率
    inclusion_probs = np.zeros(n_vars)
    for i, (var_indices, weight) in enumerate(zip(candidate_models, weights)):
        if weight > BMA_MIN_PROB / len(candidate_models):  # 忽略极低概率模型
            for idx in var_indices:
                inclusion_probs[idx] += weight

    # 创建结果字典
    var_probs = {var_names[i]: inclusion_probs[i] for i in range(n_vars)}

    return inclusion_probs, var_probs, weights, bics, model_info


# ----------------------
# 2. 数据加载与预处理
# ----------------------
print("正在加载数据...")
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

# 核心变量：RV（对数化）
rv = df_clean['rv'].values
n_samples = len(rv)
rv_log = np.log1p(rv)  # 避免RV=0

# 生成RV的三层滞后项
rv_lags = generate_all_lags(rv_log, n_samples)

# 提取外生因子并生成三层滞后项
## 2.1 平均空气污染
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_all_lags(X_avgplt, n_samples)

## 2.2 加权空气污染
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_all_lags(X_weightplt, n_samples)

## 2.3 加权气象变量
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
print(f"Weather变量数量：{len(weather_cols)}")
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_all_lags(X_weather, n_samples)

## 2.4 气候变化关注度（前3主成分）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
print(f"CLMFactor变量数量：{len(clmfactor_cols)}")
X_clmfactor_raw = df_clean[clmfactor_cols].values
X_clmfactor, clm_loadings, clm_exp_var = pca_analysis(X_clmfactor_raw, clmfactor_cols, 3, 'CLMFactor')
exog_lags_clmfactor = generate_all_lags(X_clmfactor, n_samples)
clmfactor_pc_names = [f'CLMFactor_PC{i + 1}' for i in range(3)]

## 2.5 极端天气关注度（前3主成分）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
X_weafactor, wea_loadings, wea_exp_var = pca_analysis(X_weafactor_raw, weafactor_cols, 3, 'WEAFactor')
exog_lags_weafactor = generate_all_lags(X_weafactor, n_samples)
weafactor_pc_names = [f'WEAFactor_PC{i + 1}' for i in range(3)]

## 2.6 所有外生因子
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_all_lags(X_all, n_samples)
all_exog_names = avgplt_cols + weightplt_cols + weather_cols + clmfactor_pc_names + weafactor_pc_names

# 构建模型字典（只包含6个衍生模型，排除基准HAR）
models_dict = {
    "HAR-avgplt": (rv_lags, exog_lags_avgplt, avgplt_cols),
    "HAR-weightplt": (rv_lags, exog_lags_weightplt, weightplt_cols),
    "HAR-weather": (rv_lags, exog_lags_weather, weather_cols),
    "HAR-clmfactor": (rv_lags, exog_lags_clmfactor, clmfactor_pc_names),
    "HAR-weafactor": (rv_lags, exog_lags_weafactor, weafactor_pc_names),
    "HAR-all": (rv_lags, exog_lags_all, all_exog_names)
}

# 去除前22天缺失值
valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测（BMA版本）- 使用增强版BMA函数
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)

# 存储预测结果
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': [],
                                'bma_weights': [], 'selected_vars': []}
                            for h in forecast_horizons}
                    for model in models_dict.keys()}

# 存储BMA详细信息
bma_details_dict = {model: {h: [] for h in forecast_horizons} for model in models_dict.keys()}

# 新增：存储模型后验概率和BIC信息用于可视化
model_probability_info = {model: {h: [] for h in forecast_horizons} for model in models_dict.keys()}
bic_distribution_info = {model: {h: [] for h in forecast_horizons} for model in models_dict.keys()}

print("\n开始BMA滚动窗口预测...")
start_time = time.time()

for h in forecast_horizons:
    print(f"\n正在执行{h}步超前BMA预测...")
    # 目标变量：未来h天平均对数RV
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])

    # 滚动预测
    total_windows = len(y_target) - window_size
    for i in range(window_size, len(y_target)):
        if i % 50 == 0:
            print(f"  处理进度：{i - window_size + 1}/{total_windows}")

        # 样本索引
        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        test_idx_raw = valid_start + i
        pred_date = dates_valid[i + h - 1]
        y_train = np.nan_to_num(y_target[i - window_size: i], nan=np.nanmean(y_target))
        true_rv = np.expm1(y_target[i])

        # 逐个模型处理
        for model_name, (rv_lags, exog_lags, exog_var_names) in models_dict.items():
            # 构建完整自变量矩阵（包含所有滞后项）
            X_full, var_names_full = build_model_matrix_and_names(
                rv_lags, exog_lags, exog_var_names, h, include_rv=True
            )

            # 样本内数据
            X_train = X_full[train_idx_raw]
            X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
            X_test = X_full[test_idx_raw].reshape(1, -1)
            X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))

            # BMA变量选择 - 使用增强版函数
            try:
                inclusion_probs, var_probs, weights, bics, model_info = calculate_bma_weights_enhanced(
                    X_train, y_train, var_names_full
                )

                # 保存模型概率和BIC信息（只保存第一个窗口的用于可视化）
                if len(model_probability_info[model_name][h]) < 5:  # 保存前5个窗口的数据
                    model_probability_info[model_name][h].append({
                        'date': pred_date,
                        'weights': weights[:TOP_N_MODELS],  # 前N个模型的权重
                        'bics': bics[:TOP_N_MODELS],  # 前N个模型的BIC
                        'model_info': model_info[:TOP_N_MODELS]  # 前N个模型的信息
                    })

                # 选择后验概率>0.3的变量
                selected_indices = np.where(inclusion_probs > 0.3)[0]

                if len(selected_indices) == 0:
                    # 如果没有变量被选中，使用所有变量
                    selected_indices = np.arange(len(var_names_full))
                    selected_probs = inclusion_probs
                else:
                    selected_probs = inclusion_probs[selected_indices]

                # 使用选中的变量进行预测
                X_train_selected = X_train[:, selected_indices]
                X_test_selected = X_test[:, selected_indices]

                # 训练最终模型
                final_model = LinearRegression()
                final_model.fit(X_train_selected, y_train)
                y_pred_log = final_model.predict(X_test_selected)[0]

                # 记录BMA权重信息
                bma_info = {
                    'date': pred_date,
                    'all_vars': var_names_full,
                    'inclusion_probs': inclusion_probs.tolist(),
                    'selected_vars': [var_names_full[idx] for idx in selected_indices],
                    'selected_probs': selected_probs.tolist(),
                    'n_selected': len(selected_indices),
                    'weights': weights.tolist() if len(weights) > 0 else [],
                    'bics': bics.tolist() if len(bics) > 0 else []
                }
                bma_details_dict[model_name][h].append(bma_info)

            except Exception as e:
                print(f"    BMA计算失败，使用全变量模型：{str(e)}")
                # BMA失败时使用全变量模型
                final_model = LinearRegression()
                final_model.fit(X_train, y_train)
                y_pred_log = final_model.predict(X_test)[0]
                bma_details_dict[model_name][h].append({
                    'date': pred_date,
                    'all_vars': var_names_full,
                    'inclusion_probs': [1.0 / len(var_names_full)] * len(var_names_full),
                    'selected_vars': var_names_full,
                    'selected_probs': [1.0 / len(var_names_full)] * len(var_names_full),
                    'n_selected': len(var_names_full),
                    'weights': [],
                    'bics': [],
                    'error': str(e)
                })

            y_pred_rv = np.expm1(y_pred_log)

            # 记录预测结果
            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

elapsed_time = time.time() - start_time
print(f"\nBMA预测完成，耗时：{elapsed_time:.2f}秒")

# ----------------------
# 4. 新增可视化：模型后验概率分布和BIC值分布
# ----------------------

# 4.7 可视化前10个最优模型的权重分布
for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]

    for model_name in models_dict.keys():
        if len(model_probability_info[model_name][h]) > 0:
            # 取第一个窗口的数据
            sample_info = model_probability_info[model_name][h][0]
            weights = sample_info['weights']
            model_infos = sample_info['model_info']

            if len(weights) > 0:
                # 创建模型名称列表
                model_names_display = []
                for info in model_infos:
                    if len(info['var_names']) == 0:
                        model_names_display.append('空模型')
                    else:
                        # 只显示前3个变量名
                        vars_short = ', '.join(info['var_names'][:3])
                        if len(info['var_names']) > 3:
                            vars_short += '...'
                        model_names_display.append(vars_short)

                # 绘制前10个模型的权重分布
                plt.figure(figsize=(12, 6))
                x_pos = np.arange(len(weights))
                bars = plt.bar(x_pos, weights, alpha=0.7, color='steelblue')

                # 标记最高权重
                max_idx = np.argmax(weights)
                bars[max_idx].set_color('red')

                plt.xlabel('模型')
                plt.ylabel('后验概率')
                plt.title(f'{model_name} - {horizon_name} 前{len(weights)}个最优模型后验概率分布')
                plt.xticks(x_pos, model_names_display, rotation=45, ha='right', fontsize=8)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(f'{fig_save_path}/{model_name}_{horizon_name}_模型后验概率.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

# 4.8 可视化BIC值分布
for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]

    for model_name in models_dict.keys():
        if len(model_probability_info[model_name][h]) > 0:
            # 收集所有保存窗口的BIC值
            all_bics = []
            for info in model_probability_info[model_name][h]:
                all_bics.extend(info['bics'])

            if len(all_bics) > 0:
                plt.figure(figsize=(12, 6))

                # 绘制直方图
                plt.subplot(1, 2, 1)
                plt.hist(all_bics, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                plt.xlabel('BIC值')
                plt.ylabel('频次')
                plt.title(f'{model_name} - {horizon_name} BIC值分布')
                plt.grid(True, alpha=0.3)

                # 绘制箱线图
                plt.subplot(1, 2, 2)
                plt.boxplot(all_bics, vert=True)
                plt.ylabel('BIC值')
                plt.title(f'{model_name} - {horizon_name} BIC箱线图')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'{fig_save_path}/{model_name}_{horizon_name}_BIC分布.png', dpi=300, bbox_inches='tight')
                plt.close()

# 4.9 可视化各模型在第一个窗口的BIC比较
for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]

    plt.figure(figsize=(15, 8))
    bic_data = []
    model_labels = []

    for model_name in models_dict.keys():
        if len(model_probability_info[model_name][h]) > 0:
            # 取第一个窗口的BIC值
            sample_bics = model_probability_info[model_name][h][0]['bics']
            if len(sample_bics) > 0:
                bic_data.append(sample_bics[:20])  # 取前20个
                model_labels.append(model_name)

    if len(bic_data) > 0:
        # 绘制分组箱线图
        positions = np.arange(len(model_labels)) * 3
        bp = plt.boxplot(bic_data, positions=positions, widths=0.6, patch_artist=True)

        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])

        plt.xlabel('模型')
        plt.ylabel('BIC值')
        plt.title(f'{horizon_name} - 各模型前20个候选模型BIC对比')
        plt.xticks(positions, model_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}/{horizon_name}_各模型BIC对比.png', dpi=300, bbox_inches='tight')
        plt.close()

# 4.10 可视化累计后验概率
for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]

    for model_name in models_dict.keys():
        if len(model_probability_info[model_name][h]) > 0:
            # 取第一个窗口的权重
            weights = model_probability_info[model_name][h][0]['weights']

            if len(weights) > 0:
                # 计算累计概率
                cumulative_probs = np.cumsum(weights)

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(cumulative_probs) + 1), cumulative_probs, 'b-o', linewidth=2, markersize=4)
                plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%阈值')
                plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80%阈值')
                plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%阈值')

                plt.xlabel('模型数量')
                plt.ylabel('累计后验概率')
                plt.title(f'{model_name} - {horizon_name} 累计后验概率')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{fig_save_path}/{model_name}_{horizon_name}_累计概率.png', dpi=300, bbox_inches='tight')
                plt.close()

# ----------------------
# 5. 结果整理与可视化（原有部分）
# ----------------------
# 5.1 统计指标
stats_results = []
for model_name in models_dict.keys():
    for h_idx, h in enumerate(forecast_horizons):
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        # 计算平均选中变量数
        avg_selected = np.mean([info['n_selected'] for info in bma_details_dict[model_name][h]])

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[h_idx],
            "有效样本数": valid_len,
            "MSE": round(mse, 8),
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6),
            "平均选中变量数": round(avg_selected, 2)
        })
stats_df = pd.DataFrame(stats_results)

# 5.2 变量包含概率汇总
inclusion_prob_results = []
for model_name in models_dict.keys():
    for h_idx, h in enumerate(forecast_horizons):
        # 收集所有变量的包含概率
        all_vars_dict = {}

        for info in bma_details_dict[model_name][h]:
            for var, prob in zip(info['all_vars'], info['inclusion_probs']):
                if var not in all_vars_dict:
                    all_vars_dict[var] = []
                all_vars_dict[var].append(prob)

        for var, probs in all_vars_dict.items():
            avg_prob = np.mean(probs) * 100
            inclusion_prob_results.append({
                "模型名称": model_name,
                "预测步数": horizon_names[h_idx],
                "变量名称": var,
                "平均包含概率(%)": round(avg_prob, 2),
                "最大包含概率(%)": round(np.max(probs) * 100, 2),
                "最小包含概率(%)": round(np.min(probs) * 100, 2)
            })
inclusion_prob_df = pd.DataFrame(inclusion_prob_results)

# 5.3 变量选择频率
selection_freq_results = []
for model_name in models_dict.keys():
    for h_idx, h in enumerate(forecast_horizons):
        selected_counts = {}
        total_windows = len(bma_details_dict[model_name][h])

        for info in bma_details_dict[model_name][h]:
            for var in info['selected_vars']:
                selected_counts[var] = selected_counts.get(var, 0) + 1

        for var, count in selected_counts.items():
            freq = count / total_windows * 100 if total_windows > 0 else 0
            selection_freq_results.append({
                "模型名称": model_name,
                "预测步数": horizon_names[h_idx],
                "变量名称": var,
                "选中次数": count,
                "总预测次数": total_windows,
                "选中频率(%)": round(freq, 2)
            })
selection_freq_df = pd.DataFrame(selection_freq_results)

# 5.4 可视化预测结果（原有）
for model_name in models_dict.keys():
    plt.figure(figsize=(15, 10))
    for h_idx, h in enumerate(forecast_horizons):
        dates = predictions_dict[model_name][h]['dates']
        pred = predictions_dict[model_name][h]['pred']
        true = predictions_dict[model_name][h]['true']

        plt.subplot(3, 1, h_idx + 1)
        plt.plot(dates, true, label='真实值', color='blue', linewidth=1)
        plt.plot(dates, pred, label='BMA预测值', color='red', linewidth=1, alpha=0.7)
        plt.title(f'{model_name} - {horizon_names[h_idx]} (BMA)', fontsize=12)
        plt.xlabel('日期')
        plt.ylabel('RV')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{model_name}_BMA预测结果.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5.5 可视化变量包含概率（热力图）
for h_idx, h in enumerate(forecast_horizons):
    plt.figure(figsize=(14, 10))

    # 准备数据
    plot_data = inclusion_prob_df[inclusion_prob_df['预测步数'] == horizon_names[h_idx]]

    # 选择每个模型前10个最重要变量
    top_vars_by_model = []
    for model_name in models_dict.keys():
        model_data = plot_data[plot_data['模型名称'] == model_name]
        top_vars = model_data.nlargest(10, '平均包含概率(%)')['变量名称'].tolist()
        top_vars_by_model.extend(top_vars)

    # 去重
    unique_top_vars = list(dict.fromkeys(top_vars_by_model))[:15]  # 最多显示15个变量

    # 筛选数据
    plot_data_filtered = plot_data[plot_data['变量名称'].isin(unique_top_vars)]

    # 创建透视表
    pivot_data = plot_data_filtered.pivot_table(
        values='平均包含概率(%)',
        index='变量名称',
        columns='模型名称',
        aggfunc='first'
    )

    # 绘制热力图
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': '平均包含概率(%)'})
    plt.title(f'{horizon_names[h_idx]} - 变量BMA包含概率热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{horizon_names[h_idx]}_变量包含概率热力图.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5.6 可视化变量选中频率（原有）
for h_idx, h in enumerate(forecast_horizons):
    plt.figure(figsize=(14, 8))
    subset = selection_freq_df[selection_freq_df['预测步数'] == horizon_names[h_idx]]

    # 选择每个模型前5个最重要变量
    top_vars = []
    for model_name in models_dict.keys():
        model_data = subset[subset['模型名称'] == model_name]
        top_vars.extend(model_data.nlargest(5, '选中频率(%)')['变量名称'].tolist())

    top_vars_unique = list(dict.fromkeys(top_vars))[:15]
    plot_data = subset[subset['变量名称'].isin(top_vars_unique)]

    if len(plot_data) > 0:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='选中频率(%)', y='变量名称', hue='模型名称', data=plot_data)
        plt.title(f'{horizon_names[h_idx]} - 变量选中频率（BMA）', fontsize=14)
        plt.xlabel('选中频率(%)')
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}/{horizon_names[h_idx]}_BMA变量选中频率.png', dpi=300, bbox_inches='tight')
        plt.close()

# ----------------------
# 6. 结果保存
# ----------------------
print(f"\n正在保存结果到：{output_path}")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 统计指标
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)

    # BMA包含概率
    inclusion_prob_df.to_excel(writer, sheet_name="变量包含概率", index=False)

    # 变量选择频率
    selection_freq_df.to_excel(writer, sheet_name="变量选择频率", index=False)

    # PCA结果
    clm_loadings.to_excel(writer, sheet_name="CLMFactor_PCA载荷")
    clm_exp_var.to_excel(writer, sheet_name="CLMFactor_PCA贡献率", index=False)
    wea_loadings.to_excel(writer, sheet_name="WEAFactor_PCA载荷")
    wea_exp_var.to_excel(writer, sheet_name="WEAFactor_PCA贡献率", index=False)

    # 预测值（每个步数一个sheet）
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        # 创建基础DataFrame
        result_df = pd.DataFrame({
            "日期": predictions_dict[list(models_dict.keys())[0]][h]['dates']
        })

        for model_name in models_dict.keys():
            result_df[f"{model_name}_预测RV"] = [round(val, 6) for val in predictions_dict[model_name][h]['pred']]
            result_df[f"{model_name}_真实RV"] = [round(val, 6) for val in predictions_dict[model_name][h]['true']]

            # 添加选中变量数
            selected_counts = [info['n_selected'] for info in bma_details_dict[model_name][h]]
            result_df[f"{model_name}_选中变量数"] = selected_counts

        result_df.to_excel(writer, sheet_name=f"{horizon_name}预测值", index=False)

    # BMA详细信息（可选）
    bma_summary = []
    for model_name in models_dict.keys():
        for h_idx, h in enumerate(forecast_horizons):
            for info in bma_details_dict[model_name][h]:
                bma_summary.append({
                    '模型名称': model_name,
                    '预测步数': horizon_names[h_idx],
                    '日期': info['date'],
                    '选中变量': ', '.join(info['selected_vars'][:5]) + (
                        '...' if len(info['selected_vars']) > 5 else ''),
                    '选中数量': info['n_selected']
                })
    if bma_summary:
        pd.DataFrame(bma_summary).to_excel(writer, sheet_name="BMA选择摘要", index=False)

# ----------------------
# 7. 输出汇总信息
# ----------------------
print(f"\n✅ 所有结果已保存至：{output_path}")
print(f"📊 可视化结果已保存至：{fig_save_path}")
print("\n📊 统计指标汇总（BMA模型）：")
print(stats_df[["模型名称", "预测步数", "MSE", "QLIKE", "平均选中变量数"]].head(10))

# 输出最佳模型
print("\n🏆 各预测步数最佳BMA模型（按MSE）：")
for h_idx, h in enumerate(forecast_horizons):
    h_stats = stats_df[stats_df['预测步数'] == horizon_names[h_idx]]
    best_model = h_stats.loc[h_stats['MSE'].idxmin()]
    print(
        f"{horizon_names[h_idx]}: {best_model['模型名称']} (MSE={best_model['MSE']:.6f}, 平均选中变量={best_model['平均选中变量数']:.1f})")

# 输出高频选中变量
print("\n📋 BMA高频选中变量（频率>50%）：")
high_freq_vars = selection_freq_df[selection_freq_df['选中频率(%)'] > 50]
for model_name in models_dict.keys():
    model_high_freq = high_freq_vars[high_freq_vars['模型名称'] == model_name]
    if not model_high_freq.empty:
        print(f"\n{model_name}：")
        print(model_high_freq[['预测步数', '变量名称', '选中频率(%)']].head(10).to_string(index=False))

# 输出平均变量选择数量
print("\n📊 BMA变量选择统计：")
for model_name in models_dict.keys():
    print(f"\n{model_name}:")
    for h_idx, h in enumerate(forecast_horizons):
        avg_selected = np.mean([info['n_selected'] for info in bma_details_dict[model_name][h]])
        total_vars = len(bma_details_dict[model_name][h][0]['all_vars']) if bma_details_dict[model_name][h] else 0
        print(f"  {horizon_names[h_idx]}: 平均选中 {avg_selected:.1f}/{total_vars} 个变量")

# 输出新增的可视化信息摘要
print("\n📊 新增可视化结果：")
print("  - 模型后验概率分布图（前10个最优模型）")
print("  - BIC值分布直方图与箱线图")
print("  - 各模型BIC对比箱线图")
print("  - 累计后验概率图")

print("\n✅ BMA模型分析完成！")