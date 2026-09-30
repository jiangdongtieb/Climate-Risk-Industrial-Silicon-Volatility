import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置（严格匹配论文设定）
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"..\..\results\Enet-HAR模型结果.xlsx"

# 预测配置（论文2.5节：1步=日度，5步=周度，22步=月度）
forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
# 映射：预测步数→需纳入的滞后项类型（核心逻辑）
horizon_lag_map = {
    1: ["daily"],  # 1步：仅日度（前1期）
    5: ["daily", "weekly"],  # 5步：日度（前1期）+ 周度（前5期均值）
    22: ["daily", "weekly", "monthly"]  # 22步：日度+周度+月度（前22期均值）
}

# Enet核心参数（论文2.4.4节明确设定）
ENET_MU = 0.5  # 平衡L1/L2惩罚，论文固定为0.5
CV_FOLDS = 5  # 时间序列交叉验证折数（贴合论文交叉验证逻辑）
MAX_ITER = 10000  # 最大迭代次数（确保收敛）


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
# 核心2：按步数构建自变量矩阵（适配Enet变量选择）
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

    # 3. 合并RV+外生变量（Enet模型的完整自变量矩阵）
    return np.hstack([rv_matrix, exog_matrix])


# ----------------------
# 核心3：Enet变量选择与系数估计（适配动态自变量矩阵）
# ----------------------
def enet_coefficient_estimation(X, y):
    """
    论文适配版Enet系数估计：
    1. 标准化（论文要求Enet基于标准化变量建模）
    2. 时间序列5折交叉验证选择λ（避免数据泄露）
    3. 固定μ=0.5（论文明确设定）
    """
    # 标准化（Enet对尺度敏感，论文强制要求）
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 时间序列交叉验证（5折，贴合论文滚动预测逻辑，避免数据泄露）
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    # Enet交叉验证（固定μ=0.5，论文2.4.4节定义）
    enet_cv = ElasticNetCV(
        cv=tscv,
        l1_ratio=ENET_MU,  # 固定μ=0.5，平衡L1和L2惩罚
        n_alphas=100,  # 100个候选λ，贴合论文GLMNET实现
        max_iter=MAX_ITER,
        tol=1e-3,
        random_state=42
    )
    enet_cv.fit(X_scaled, y_scaled)

    # 最优λ与非零系数变量（Enet稀疏性特征）
    best_lambda = enet_cv.alpha_
    non_zero_idx = np.where(enet_cv.coef_ != 0)[0]
    enet_coef = enet_cv.coef_

    # 输出筛选信息（便于论文报告）
    print(f"Enet变量筛选：原始变量数{X.shape[1]} → 非零系数变量数{len(non_zero_idx)}")
    print(f"最优惩罚系数λ：{best_lambda:.6f}")
    top5_coef = np.abs(enet_coef[non_zero_idx[:5]]) if len(non_zero_idx) >= 5 else np.abs(enet_coef[non_zero_idx])
    coef_str = ", ".join([f"{val:.4f}" for val in top5_coef])
    print(f"非零系数绝对值（前{len(top5_coef)}个）：{coef_str}")

    return enet_coef, scaler_X, scaler_y, best_lambda, len(non_zero_idx)


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
## 2.1 平均空气污染变量（6个，论文表2定义）
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_all_lags(X_avgplt, n_samples)

## 2.2 加权空气污染变量（3个，论文表2定义）
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_all_lags(X_weightplt, n_samples)

## 2.3 加权气象变量（5个，论文表2定义）
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_all_lags(X_weather, n_samples)

## 2.4 气候变化关注度（前3主成分，论文2.2.4节）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
pca_clm = PCA(n_components=3, random_state=42)
X_clmfactor = pca_clm.fit_transform(X_clmfactor_raw)
exog_lags_clmfactor = generate_all_lags(X_clmfactor, n_samples)

## 2.5 极端天气关注度（前3主成分，论文2.2.5节）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
pca_wea = PCA(n_components=3, random_state=42)
X_weafactor = pca_wea.fit_transform(X_weafactor_raw)
exog_lags_weafactor = generate_all_lags(X_weafactor, n_samples)

## 2.6 所有外生因子（Enet-HAR-all模型用）
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_all_lags(X_all, n_samples)

# 构建6个模型的滞后项字典（便于按步数调用）
models_lags = {
    "Enet-HAR-avgplt": (rv_lags, exog_lags_avgplt),
    "Enet-HAR-weightplt": (rv_lags, exog_lags_weightplt),
    "Enet-HAR-weather": (rv_lags, exog_lags_weather),
    "Enet-HAR-clmfactor": (rv_lags, exog_lags_clmfactor),
    "Enet-HAR-weafactor": (rv_lags, exog_lags_weafactor),
    "Enet-HAR-all": (rv_lags, exog_lags_all)
}

# 去除前22天缺失值（月度滞后项导致，论文3节样本处理）
valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测（核心修复：修正enet_info字典结构）
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)  # 样本内占2/3，最小22天
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons} for model in
                    models_lags.keys()}
# 核心修复：按「模型名称→预测步数」初始化字典（与调用逻辑一致）
enet_info = {}
for model_name in models_lags.keys():
    enet_info[model_name] = {}
    for h in forecast_horizons:
        enet_info[model_name][h] = {}

for h in forecast_horizons:
    print(f"\n=== 开始{h}步超前预测 ===")
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

        # 训练Enet-HAR模型并预测（论文核心逻辑）
        for model_name, X_train in X_train_dict.items():
            # 步骤1：Enet系数估计（仅首次滚动执行，后续复用）
            if len(enet_info[model_name][h]) == 0:
                enet_coef, scaler_X, scaler_y, best_lambda, non_zero_cnt = enet_coefficient_estimation(X_train, y_train)
                enet_info[model_name][h] = {
                    "enet_coef": enet_coef,
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "best_lambda": best_lambda,
                    "non_zero_cnt": non_zero_cnt,
                    "total_vars": X_train.shape[1]
                }
            else:
                enet_coef = enet_info[model_name][h]["enet_coef"]
                scaler_X = enet_info[model_name][h]["scaler_X"]
                scaler_y = enet_info[model_name][h]["scaler_y"]

            # 步骤2：标准化测试数据（用样本内标准化器）
            X_test_scaled = scaler_X.transform(X_test_dict[model_name])
            # 步骤3：计算对数预测值（y_pred_log = X_test_scaled @ enet_coef）
            y_pred_log_scaled = np.dot(X_test_scaled, enet_coef)[0]
            # 步骤4：逆标准化还原为原始对数RV尺度
            y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1))[0][0]
            # 步骤5：反对数化还原为原始RV
            y_pred_rv = np.expm1(y_pred_log)

            # 记录结果
            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

# ----------------------
# 4. 统计指标与结果保存（MSE保留8位小数，贴合论文表格）
# ----------------------
stats_results = []
for model_name in models_lags.keys():
    for h_idx, h in enumerate(forecast_horizons):
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        # 提取Enet信息
        info = enet_info[model_name][h]
        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[h_idx],
            "有效样本数": valid_len,
            "总变量数": info["total_vars"],
            "非零变量数": info["non_zero_cnt"],
            "最优λ": round(info["best_lambda"], 6),
            "MSE": round(mse, 8),  # 核心：MSE保留8位小数
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# Enet模型信息汇总（贴合论文表格格式）
enet_summary = []
for model_name in models_lags.keys():
    for h in forecast_horizons:
        info = enet_info[model_name][h]
        enet_summary.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "总变量数": info["total_vars"],
            "非零变量数": info["non_zero_cnt"],
            "变量筛选比例": round(info["non_zero_cnt"] / info["total_vars"], 4),
            "最优惩罚系数λ": round(info["best_lambda"], 6),
            "L1/L2平衡参数μ": ENET_MU,
            "纳入滞后项类型": ", ".join(horizon_lag_map[h])
        })
enet_summary_df = pd.DataFrame(enet_summary)

# 保存结果（多工作表，符合论文输出格式）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 工作表1：统计指标汇总（8位小数，可计算相对损失）
    stats_df.to_excel(writer, sheet_name="统计指标汇总（8位小数）", index=False)
    # 工作表2：Enet模型配置信息
    enet_summary_df.to_excel(writer, sheet_name="Enet配置信息", index=False)
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
print("\n=== Enet-HAR模型配置汇总 ===")
print(enet_summary_df[["模型名称", "预测步数", "总变量数", "非零变量数", "纳入滞后项类型"]].head(12))
print(f"\n📊 统计指标汇总（前10行）：")
print(stats_df[["模型名称", "预测步数", "MSE", "QLIKE"]].head(10))
# ----------------------
# 1. 全局配置（严格匹配论文设定）
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\Enet-HAR模型预测结果及统计指标.xlsx"

# 预测配置（论文2.5节：1步=日度，5步=周度，22步=月度）
forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]

# Enet核心参数（论文2.4.4节明确设定）
ENET_MU = 0.5  # 平衡L1/L2惩罚，论文固定为0.5
CV_FOLDS = 5  # 时间序列交叉验证折数（贴合论文交叉验证逻辑）
MAX_ITER = 10000  # 最大迭代次数（确保收敛）


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
# 核心：Enet变量选择与系数估计（严格复刻论文2.4.4节）
# ----------------------
def enet_coefficient_estimation(X, y):
    """
    论文适配版Enet系数估计：
    1. 标准化（论文要求Enet基于标准化变量建模）
    2. 时间序列5折交叉验证选择λ（避免数据泄露）
    3. 固定μ=0.5（论文明确设定）
    """
    # 标准化（Enet对尺度敏感，论文强制要求）
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 时间序列交叉验证（5折，贴合论文滚动预测逻辑，避免数据泄露）
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    # Enet交叉验证（固定μ=0.5，论文2.4.4节定义）
    enet_cv = ElasticNetCV(
        cv=tscv,
        l1_ratio=ENET_MU,  # 固定μ=0.5，平衡L1和L2惩罚
        n_alphas=100,  # 100个候选λ，贴合论文GLMNET实现
        max_iter=MAX_ITER,
        tol=1e-3,
        random_state=42
    )
    enet_cv.fit(X_scaled, y_scaled)

    # 最优λ与非零系数变量（Enet稀疏性特征）
    best_lambda = enet_cv.alpha_
    non_zero_idx = np.where(enet_cv.coef_ != 0)[0]
    enet_coef = enet_cv.coef_

    # 输出筛选信息（便于论文报告）
    print(f"Enet变量筛选：原始变量数{X.shape[1]} → 非零系数变量数{len(non_zero_idx)}")
    print(f"最优惩罚系数λ：{best_lambda:.6f}")
    top5_coef = np.abs(enet_coef[non_zero_idx[:5]]) if len(non_zero_idx) >= 5 else np.abs(enet_coef[non_zero_idx])
    coef_str = ", ".join([f"{val:.4f}" for val in top5_coef])
    print(f"非零系数绝对值（前{len(top5_coef)}个）：{coef_str}")

    return enet_coef, scaler_X, scaler_y, best_lambda, len(non_zero_idx)


# ----------------------
# 2. 数据加载与预处理（严格匹配论文2.2节变量定义）
# ----------------------
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

# 核心变量：RV（已实现波动率），对数化处理（论文公式5-11）
rv = df_clean['rv'].values
rv_log = np.log1p(rv)  # log1p = log(x+1)，避免RV=0

# 构建HAR核心滞后项（日度、周度、月度，论文2.3节Model 1）
rv_lag1_log = np.roll(rv_log, 1)  # 滞后1天（log(RV_{t-1})）
rv_lag5_log = np.convolve(rv_log, np.ones(5) / 5, mode='same')  # 滞后5天平均（周度）
rv_lag22_log = np.convolve(rv_log, np.ones(22) / 22, mode='same')  # 滞后22天平均（月度）
HAR_core = np.vstack([rv_lag1_log, rv_lag5_log, rv_lag22_log]).T  # 3个核心变量

# 提取外生因子（严格按论文2.2节5组因子，不含多晶硅相关变量）
## 2.1 平均空气污染变量（6个，论文表2定义）
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values

## 2.2 加权空气污染变量（3个，论文表2定义）
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values

## 2.3 加权气象变量（5个，论文表2定义）
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values

## 2.4 气候变化关注度（前3主成分，论文2.2.4节）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
pca_clm = PCA(n_components=3, random_state=42)
X_clmfactor = pca_clm.fit_transform(X_clmfactor_raw)

## 2.5 极端天气关注度（前3主成分，论文2.2.5节）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
pca_wea = PCA(n_components=3, random_state=42)
X_weafactor = pca_wea.fit_transform(X_weafactor_raw)

## 2.6 所有外生因子（Enet-HAR-all模型用）
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])

# 构建6个Enet-HAR模型的自变量矩阵（论文2.3节Model 2-7）
models_raw = {
    "Enet-HAR-avgplt": np.hstack([HAR_core, X_avgplt]),  # 3+6=9个变量
    "Enet-HAR-weightplt": np.hstack([HAR_core, X_weightplt]),  # 3+3=6个变量
    "Enet-HAR-weather": np.hstack([HAR_core, X_weather]),  # 3+5=8个变量
    "Enet-HAR-clmfactor": np.hstack([HAR_core, X_clmfactor]),  # 3+3=6个变量
    "Enet-HAR-weafactor": np.hstack([HAR_core, X_weafactor]),  # 3+3=6个变量
    "Enet-HAR-all": np.hstack([HAR_core, X_all])  # 3+(6+3+5+3+3)=23个变量
}

# 去除前22天缺失值（月度滞后项导致，论文3节样本处理）
valid_start = 22
for model_name, X in models_raw.items():
    models_raw[model_name] = X[valid_start:]
rv_log_valid = rv_log[valid_start:]
dates_valid = df_clean.index[valid_start:]
n_valid = len(rv_log_valid)

# 最终NaN防护
for model_name, X in models_raw.items():
    models_raw[model_name] = np.nan_to_num(X, nan=0)

# ----------------------
# 3. 滚动窗口预测（论文2.5节：2:1样本内/外比例）
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)  # 样本内占2/3，最小22天
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons} for model in
                    models_raw.keys()}
enet_info = {model: [] for model in models_raw.keys()}  # 存储每个模型的λ和非零变量数

for h in forecast_horizons:
    print(f"\n正在执行{h}步超前预测...")
    # 目标变量：未来h天平均对数RV（直接预测法，论文2.5节）
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log_valid[i:i + h])

    # 滚动预测
    for i in range(window_size, len(y_target)):
        train_idx = range(i - window_size, i)
        test_idx = i
        pred_date = dates_valid[i + h - 1]  # 预测日期：未来h天最后一天（论文日期对齐）

        # 样本内数据
        X_train_dict = {}
        y_train = y_target[train_idx]

        for model_name, X in models_raw.items():
            X_train = X[train_idx]
            X_train_dict[model_name] = X_train

        # 样本外数据
        X_test_dict = {}
        for model_name, X in models_raw.items():
            X_test = X[test_idx].reshape(1, -1)
            X_test_dict[model_name] = X_test

        # 真实值（反对数化还原为原始RV尺度）
        true_rv = np.expm1(y_target[test_idx])

        # 训练Enet-HAR模型并预测（论文核心逻辑）
        for model_name, X_train in X_train_dict.items():
            # 步骤1：Enet系数估计（按论文公式计算）
            enet_coef, scaler_X, scaler_y, best_lambda, non_zero_cnt = enet_coefficient_estimation(X_train, y_train)
            # 步骤2：标准化测试数据（用样本内标准化器）
            X_test_scaled = scaler_X.transform(X_test_dict[model_name])
            # 步骤3：计算对数预测值（y_pred_log = X_test_scaled @ enet_coef）
            y_pred_log_scaled = np.dot(X_test_scaled, enet_coef)[0]
            # 步骤4：逆标准化还原为原始对数RV尺度
            y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1))[0][0]
            # 步骤5：反对数化还原为原始RV
            y_pred_rv = np.expm1(y_pred_log)

            # 记录结果
            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

            # 记录Enet信息（仅记录首次滚动的λ和非零变量数，避免重复）
            if len(enet_info[model_name]) == 0:
                enet_info[model_name] = {
                    "最优λ": best_lambda,
                    "非零变量数": non_zero_cnt,
                    "总变量数": X_train.shape[1]
                }

# ----------------------
# 4. 统计指标与结果保存（论文4.1节统计评估）
# ----------------------
# 统计指标计算
stats_results = []
for model_name in models_raw.keys():
    for h_idx, h in enumerate(forecast_horizons):
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[h_idx],
            "有效样本数": valid_len,
            "总变量数": enet_info[model_name]["总变量数"],
            "非零变量数": enet_info[model_name]["非零变量数"],
            "最优λ": round(enet_info[model_name]["最优λ"], 6),
            "MSE": round(mse, 6),
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# Enet模型信息汇总
enet_summary = []
for model_name, info in enet_info.items():
    enet_summary.append({
        "模型名称": model_name,
        "总变量数": info["总变量数"],
        "非零变量数": info["非零变量数"],
        "变量筛选比例": round(info["非零变量数"] / info["总变量数"], 4),
        "最优惩罚系数λ": round(info["最优λ"], 6),
        "L1/L2平衡参数μ": ENET_MU  # 论文固定为0.5
    })
enet_summary_df = pd.DataFrame(enet_summary)

# 保存结果（多工作表，符合论文输出格式）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 工作表1：统计指标汇总（论文表4风格）
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)
    # 工作表2：Enet模型配置信息
    enet_summary_df.to_excel(writer, sheet_name="Enet配置信息", index=False)
    # 工作表3-5：分步预测值
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        step_data = None
        for model_name in models_raw.keys():
            model_info = pd.DataFrame({
                "日期": predictions_dict[model_name][h]['dates'],
                f"{model_name}_预测RV": predictions_dict[model_name][h]['pred'],
                f"{model_name}_真实RV": predictions_dict[model_name][h]['true']
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
print("\n=== Enet-HAR模型配置汇总 ===")
print(enet_summary_df[["模型名称", "总变量数", "非零变量数", "最优惩罚系数λ", "L1/L2平衡参数μ"]])
print(f"\n📊 统计指标汇总（前10行）：")
print(stats_df[["模型名称", "预测步数", "非零变量数", "MSE", "QLIKE"]].head(10))
