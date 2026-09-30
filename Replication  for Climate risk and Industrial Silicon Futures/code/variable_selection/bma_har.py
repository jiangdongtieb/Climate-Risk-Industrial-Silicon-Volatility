import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import chi2
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置与核心工具函数（严格匹配论文）
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"..\..\results\BMA-HAR模型结果.xlsx"

# 预测配置（论文2.5节：1步=日度，5步=周度，22步=月度）
forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
# 映射：预测步数→需纳入的滞后项类型（核心逻辑修正）
horizon_lag_map = {
    1: ["daily"],  # 1步：仅日度（前1期）
    5: ["daily", "weekly"],  # 5步：日度（前1期）+ 周度（前5期均值）
    22: ["daily", "weekly", "monthly"]  # 22步：日度+周度+月度（前22期均值）
}


# 统计损失函数（论文2.5.1节公式）
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


# 数据清理函数（处理百分比、字符串、缺失值）
def clean_data(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if col == 'date':
            continue
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.replace('%', '').replace('nan', np.nan)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        # 中位数填充（抗极端值，论文表3变量描述性统计验证）
        fill_val = df_clean[col].median() if df_clean[col].notna().sum() > 0 else 0
        df_clean[col] = df_clean[col].fillna(fill_val)
    return df_clean


# ----------------------
# 核心1：生成三层滞后项（日度/周度/月度）
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

    # 填充前22期缺失值（用后续有效数据均值）
    daily[:22] = np.nanmean(daily[22:], axis=0)
    weekly[:22] = np.nanmean(weekly[22:], axis=0)
    monthly[:22] = np.nanmean(monthly[22:], axis=0)

    return {"daily": daily, "weekly": weekly, "monthly": monthly}


# ----------------------
# 核心2：按步数构建自变量矩阵（适配BMA模型变量子集逻辑）
# ----------------------
def build_model_matrix(rv_lags, exog_lags, horizon):
    """按预测步数构建自变量矩阵：RV滞后项 + 外生变量滞后项（一一对应）"""
    required_lags = horizon_lag_map[horizon]

    # 1. 拼接RV滞后项（按需求选择，转为列向量）
    rv_matrix = []
    for lag_type in required_lags:
        rv_col = rv_lags[lag_type].reshape(-1, 1)
        rv_matrix.append(rv_col)
    rv_matrix = np.hstack(rv_matrix)

    # 2. 拼接外生变量滞后项（与RV滞后项类型一致）
    exog_matrix = []
    for lag_type in required_lags:
        exog_matrix.append(exog_lags[lag_type])
    exog_matrix = np.hstack(exog_matrix)

    # 3. 合并RV+外生变量（BMA模型的完整自变量矩阵）
    return np.hstack([rv_matrix, exog_matrix])


# ----------------------
# 核心3：BMA系数计算（严格复刻论文2.4.3节公式，适配动态自变量矩阵）
# ----------------------
def bma_coefficient_estimation(X, y):
    """
    论文适配版BMA系数估计：
    1. 生成变量子集（HAR核心项必选，外生因子组合可选）
    2. 按论文公式计算模型权重ω_k
    3. 加权平均得到BMA系数
    """
    n_vars = X.shape[1]
    T = len(y)

    # 步骤1：定义约束模型（仅含HAR核心变量，对应当前步数的RV滞后项）
    # 识别HAR核心变量索引（前n_lags列，n_lags=1/2/3，随步数变化）
    n_lags = len(required_lags_global)  # 全局变量传递当前步数的滞后项数量
    har_core_idx = list(range(n_lags))
    X_R = X[:, har_core_idx]
    model_R = LinearRegression()
    model_R.fit(X_R, y)
    y_pred_R = model_R.predict(X_R)
    SSR_R = np.sum((y - y_pred_R) ** 2)  # 约束模型残差平方和
    df_R = T - len(har_core_idx) - 1  # 约束模型自由度

    # 步骤2：生成变量子集（HAR核心项必选，外生因子组合可选）
    exogenous_idx = list(range(n_lags, n_vars))  # 外生因子索引（核心项之后）
    n_exo = len(exogenous_idx)
    max_subsets = 1000  # 限制最大子集数（平衡效率与逻辑）

    # 生成外生因子子集（含空集）
    subsets = []
    for k in range(0, min(n_exo, 3) + 1):  # 优先低维子集，贴合论文降维逻辑
        for combo in combinations(exogenous_idx, k):
            subsets.append(har_core_idx + list(combo))
        if len(subsets) >= max_subsets:
            break
    # 若子集数不足，补充高维子集
    if len(subsets) < 50:
        for k in range(min(n_exo, 3) + 1, min(n_exo, 5) + 1):
            for combo in combinations(exogenous_idx, k):
                subsets.append(har_core_idx + list(combo))
                if len(subsets) >= max_subsets:
                    break
            if len(subsets) >= max_subsets:
                break

    # 步骤3：计算每个子集模型的参数与权重（论文公式）
    model_weights = []
    model_coefs = []
    a_g = 1  # 论文建议a(g)=1
    g = 1 / min(T, n_vars ** 2)  # g-prior设定

    for subset in subsets:
        X_k = X[:, subset]
        p_k = len(subset)
        if p_k >= T:
            continue  # 避免参数数≥样本数

        model_k = LinearRegression()
        model_k.fit(X_k, y)
        y_pred_k = model_k.predict(X_k)
        SSR_k_U = np.sum((y - y_pred_k) ** 2)  # 无约束模型残差平方和

        # 计算模型权重ω_k（论文公式）
        numerator = (a_g ** (p_k / 2)) * ((1 + (g ** (-1)) * SSR_k_U / SSR_R) ** (-df_R / 2))
        model_weights.append(numerator)
        model_coefs.append(model_k.coef_)

    # 归一化权重
    model_weights = np.array(model_weights)
    if np.sum(model_weights) == 0:
        model_weights = np.ones_like(model_weights) / len(model_weights)
    else:
        model_weights = model_weights / np.sum(model_weights)

    # 步骤4：计算BMA加权系数（论文公式β^BMA = Σω_kβ^(k)）
    bma_coef = np.zeros(n_vars)
    for idx, (subset, coef, weight) in enumerate(zip(subsets, model_coefs, model_weights)):
        for i, var_idx in enumerate(subset):
            bma_coef[var_idx] += weight * coef[i]

    return bma_coef


# ----------------------
# 2. 数据加载与预处理（严格匹配论文2.2节变量定义）
# ----------------------
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

# 核心变量：RV（已实现波动率），对数化处理（论文公式5-11）
rv = df_clean['rv'].values
n_samples = len(rv)
rv_log = np.log1p(rv)  # log1p = log(x+1)，避免RV=0

# 生成RV的三层滞后项（daily/weekly/monthly）
rv_lags = generate_all_lags(rv_log, n_samples)

# 提取外生因子并生成三层滞后项（论文表2）
## 2.1 平均空气污染变量（6个）
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_all_lags(X_avgplt, n_samples)

## 2.2 加权空气污染变量（3个）
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_all_lags(X_weightplt, n_samples)

## 2.3 加权气象变量（5个）
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_all_lags(X_weather, n_samples)

## 2.4 气候变化关注度（前3主成分）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
pca_clm = PCA(n_components=3, random_state=42)
X_clmfactor = pca_clm.fit_transform(X_clmfactor_raw)
exog_lags_clmfactor = generate_all_lags(X_clmfactor, n_samples)

## 2.5 极端天气关注度（前3主成分）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
pca_wea = PCA(n_components=3, random_state=42)
X_weafactor = pca_wea.fit_transform(X_weafactor_raw)
exog_lags_weafactor = generate_all_lags(X_weafactor, n_samples)

## 2.6 所有外生因子（BMA-HAR-all模型用）
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_all_lags(X_all, n_samples)

# 构建6个模型的滞后项字典（便于按步数调用）
models_lags = {
    "BMA-HAR-avgplt": (rv_lags, exog_lags_avgplt),
    "BMA-HAR-weightplt": (rv_lags, exog_lags_weightplt),
    "BMA-HAR-weather": (rv_lags, exog_lags_weather),
    "BMA-HAR-clmfactor": (rv_lags, exog_lags_clmfactor),
    "BMA-HAR-weafactor": (rv_lags, exog_lags_weafactor),
    "BMA-HAR-all": (rv_lags, exog_lags_all)
}

# 去除前22天缺失值（月度滞后项导致，论文3节样本处理）
valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测（核心修正：按步数动态构建自变量矩阵）
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)  # 样本内占2/3，最小22天
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons} for model in
                    models_lags.keys()}
global required_lags_global  # 全局变量：传递当前步数的滞后项需求（供BMA函数使用）

for h in forecast_horizons:
    required_lags_global = horizon_lag_map[h]  # 记录当前步数的滞后项类型
    print(f"\n正在执行{h}步超前预测（纳入滞后项：{required_lags_global}）...")
    # 目标变量：未来h天平均对数RV（直接预测法，论文2.5节）
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])

    # 滚动预测
    for i in range(window_size, len(y_target)):
        # 样本内/外索引（转为数组，避免range操作报错）
        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        test_idx_raw = valid_start + i
        pred_date = dates_valid[i + h - 1]  # 预测日期：未来h天最后一天

        # 样本内数据（按步数构建每个模型的自变量矩阵）
        X_train_dict = {}
        y_train = y_target[i - window_size: i]

        for model_name, (rv_lags, exog_lags) in models_lags.items():
            # 构建当前步数的自变量矩阵
            X_raw = build_model_matrix(rv_lags, exog_lags, h)
            X_train = X_raw[train_idx_raw]
            X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
            X_train_dict[model_name] = X_train

        # 样本外数据（按相同规则构建）
        X_test_dict = {}
        for model_name, (rv_lags, exog_lags) in models_lags.items():
            X_raw = build_model_matrix(rv_lags, exog_lags, h)
            X_test = X_raw[test_idx_raw].reshape(1, -1)
            X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
            X_test_dict[model_name] = X_test

        # 真实值（反对数化还原为原始RV尺度）
        true_rv = np.expm1(y_target[i])

        # 训练BMA-HAR模型并预测（论文核心逻辑）
        for model_name, X_train in X_train_dict.items():
            # 步骤1：BMA系数估计（按论文公式计算权重）
            bma_coef = bma_coefficient_estimation(X_train, y_train)
            # 步骤2：计算预测值（y_pred = X_test @ bma_coef）
            y_pred_log = np.dot(X_test_dict[model_name], bma_coef)[0]
            # 步骤3：反对数化还原为原始RV
            y_pred_rv = np.expm1(y_pred_log)

            # 记录结果
            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

# ----------------------
# 4. 统计指标与结果保存（MSE保留8位小数，贴合论文表格）
# ----------------------
# 统计指标计算（MSE 8位小数）
stats_results = []
for model_name in models_lags.keys():
    for h_idx, h in enumerate(forecast_horizons):
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        # 记录变量总数（当前步数的自变量矩阵维度）
        rv_lags, exog_lags = models_lags[model_name]
        X_raw = build_model_matrix(rv_lags, exog_lags, h)
        var_total = X_raw.shape[1]
        har_core_num = len(horizon_lag_map[h])  # HAR核心变量数=滞后项类型数
        exog_num = var_total - har_core_num

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[h_idx],
            "有效样本数": valid_len,
            "变量总数": var_total,
            "HAR核心变量数": har_core_num,
            "外生因子数": exog_num,
            "MSE": round(mse, 8),  # 核心：MSE保留8位小数
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# BMA模型信息汇总
bma_summary = []
for model_name, (rv_lags, exog_lags) in models_lags.items():
    # 统计各步数的变量信息
    for h in forecast_horizons:
        X_raw = build_model_matrix(rv_lags, exog_lags, h)
        var_total = X_raw.shape[1]
        har_core_num = len(horizon_lag_map[h])
        exog_num = var_total - har_core_num
        bma_summary.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "变量总数": var_total,
            "HAR核心变量数": har_core_num,
            "外生因子数": exog_num,
            "纳入滞后项类型": ", ".join(horizon_lag_map[h])
        })
bma_summary_df = pd.DataFrame(bma_summary)

# 保存结果（多工作表，符合论文输出格式）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 工作表1：统计指标汇总（8位小数，可计算相对损失）
    stats_df.to_excel(writer, sheet_name="统计指标汇总（8位小数）", index=False)
    # 工作表2：BMA模型配置信息
    bma_summary_df.to_excel(writer, sheet_name="BMA模型配置", index=False)
    # 工作表3-5：分步预测值
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        step_data = None
        for model_name in models_lags.keys():
            model_info = pd.DataFrame({
                "日期": predictions_dict[model_name][h]['dates'],
                f"{model_name}_预测RV": [round(val, 6) for val in predictions_dict[model_name][h]['pred']],
                f"{model_name}_真实RV": [round(val, 6) for val in predictions_dict[model_name][h]['true']]
            })
            if step_data is None:
                step_data = model_info
            else:
                step_data = pd.merge(step_data, model_info, on="日期", how="outer", suffixes=('', f'_{model_name}'))

        step_data.to_excel(writer, sheet_name=f"{horizon_name}预测值", index=False)

# ----------------------
# 5. 输出汇总信息
# ----------------------
print(f"\n✅ 所有结果已保存至：{output_path}")
print("\n=== BMA-HAR模型配置汇总 ===")
print(bma_summary_df[["模型名称", "预测步数", "变量总数", "纳入滞后项类型"]].head(12))
print(f"\n📊 统计指标汇总（前10行）：")
print(stats_df[["模型名称", "预测步数", "MSE", "QLIKE"]].head(10))
