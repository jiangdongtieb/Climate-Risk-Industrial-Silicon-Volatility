import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置与核心工具函数
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"..\..\results\HAR类模型结果.xlsx"

forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
# 映射：预测步数→需纳入的滞后项类型（严格贴合你的逻辑）
horizon_lag_map = {
    1: ["daily"],  # 1步：仅日度（前1期）
    5: ["daily", "weekly"],  # 5步：日度（前1期）+ 周度（前5期均值）
    22: ["daily", "weekly", "monthly"]  # 22步：日度+周度+月度（前22期均值）
}


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


# ----------------------
# 核心修正1：生成三层滞后项（日度/周度/月度）
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
# 核心修正2：构建自变量矩阵（增加外生变量为None的判断）
# ----------------------
def build_model_matrix(rv_lags, exog_lags, horizon):
    """按预测步数构建自变量矩阵：RV滞后项 + 外生变量滞后项（一一对应）"""
    required_lags = horizon_lag_map[horizon]

    # 1. 拼接RV滞后项（按需求选择，转为列向量避免维度错误）
    rv_matrix = []
    for lag_type in required_lags:
        rv_col = rv_lags[lag_type].reshape(-1, 1)  # 确保是列向量
        rv_matrix.append(rv_col)
    rv_matrix = np.hstack(rv_matrix)

    # 2. 拼接外生变量滞后项（外生变量为None时，仅返回RV滞后项）
    if exog_lags is None:
        return rv_matrix
    exog_matrix = []
    for lag_type in required_lags:
        exog_matrix.append(exog_lags[lag_type])
    exog_matrix = np.hstack(exog_matrix)

    # 3. 合并RV+外生变量（论文模型完整自变量）
    return np.hstack([rv_matrix, exog_matrix])


# ----------------------
# 2. 数据加载与预处理（严格匹配论文）
# ----------------------
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

# 核心变量：RV（已实现波动率）+ 对数化
rv = df_clean['rv'].values
n_samples = len(rv)
rv_log = np.log1p(rv)  # 避免RV=0

# 生成RV的三层滞后项（daily/weekly/monthly）
rv_lags = generate_all_lags(rv_log, n_samples)

# 提取外生因子并生成三层滞后项（论文表2）
## 2.1 平均空气污染变量（avgplt-*）
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_all_lags(X_avgplt, n_samples)

## 2.2 加权空气污染变量（weightplt-*）
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_all_lags(X_weightplt, n_samples)

## 2.3 加权气象变量（weather-*）
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_all_lags(X_weather, n_samples)

## 2.4 气候变化关注度（clmfactor前3主成分）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
pca_clm = PCA(n_components=3, random_state=42)
X_clmfactor = pca_clm.fit_transform(X_clmfactor_raw)
exog_lags_clmfactor = generate_all_lags(X_clmfactor, n_samples)

## 2.5 极端天气关注度（weafactor前3主成分）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
pca_wea = PCA(n_components=3, random_state=42)
X_weafactor = pca_wea.fit_transform(X_weafactor_raw)
exog_lags_weafactor = generate_all_lags(X_weafactor, n_samples)

## 2.6 所有外生因子（HAR-all模型用）
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_all_lags(X_all, n_samples)

# 构建7个模型的滞后项字典（便于按步数调用）
models_lags = {
    "HAR": (rv_lags, None),  # 基准模型：仅RV滞后项（外生变量为None）
    "HAR-avgplt": (rv_lags, exog_lags_avgplt),
    "HAR-weightplt": (rv_lags, exog_lags_weightplt),
    "HAR-weather": (rv_lags, exog_lags_weather),
    "HAR-clmfactor": (rv_lags, exog_lags_clmfactor),
    "HAR-weafactor": (rv_lags, exog_lags_weafactor),
    "HAR-all": (rv_lags, exog_lags_all)
}

# 去除前22天缺失值（与论文一致）
valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测（无语法错误）
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)  # 样本内占2/3，最小22天
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons} for model in
                    models_lags.keys()}

for h in forecast_horizons:
    print(f"正在执行{h}步超前预测...")
    # 目标变量：未来h天平均对数RV（论文2.5节直接预测法）
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
        pred_date = dates_valid[i + h - 1]  # 预测日期：未来h天的最后一天

        # 样本内数据（按步数构建每个模型的自变量矩阵）
        X_train_dict = {}
        for model_name, (rv_lags, exog_lags) in models_lags.items():
            # 核心修复：调用函数时自动处理外生变量为None的情况
            X_raw = build_model_matrix(rv_lags, exog_lags, h)
            # 提取样本内数据并处理NaN
            X_train = X_raw[train_idx_raw]
            X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
            X_train_dict[model_name] = X_train

        # 样本内目标变量（处理NaN）
        y_train = np.nan_to_num(y_target[i - window_size: i], nan=np.nanmean(y_target))

        # 样本外数据（按相同规则构建）
        X_test_dict = {}
        for model_name, (rv_lags, exog_lags) in models_lags.items():
            X_raw = build_model_matrix(rv_lags, exog_lags, h)
            X_test = X_raw[test_idx_raw].reshape(1, -1)
            X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
            X_test_dict[model_name] = X_test

        # 真实值（反对数化还原为原始RV尺度）
        true_rv = np.expm1(y_target[i])

        # 训练并预测每个模型
        for model_name, X_train in X_train_dict.items():
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test_dict[model_name])[0]
            y_pred_rv = np.expm1(y_pred_log)

            # 记录结果（日期、预测值、真实值一一对应）
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
        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[h_idx],
            "有效样本数": valid_len,
            "MSE": round(mse, 8),  # 核心：MSE保留8位小数
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# 保存结果（多工作表，与论文输出格式一致）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 工作表1：统计指标汇总（可直接计算相对损失匹配论文表格）
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)
    # 工作表2-4：按步数拆分的预测值
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
print("\n📊 统计指标汇总（前10行）：")
print(stats_df[["模型名称", "预测步数", "MSE", "QLIKE"]].head(10))

# 额外输出：各模型在不同步数的原始变量数（验证滞后项匹配逻辑）
print("\n📋 各模型原始变量数（验证滞后项逻辑）：")
for model_name, (rv_lags, exog_lags) in models_lags.items():
    if model_name == "HAR":
        exog_dim = 0
    else:
        exog_dim = exog_lags["daily"].shape[1]
    for h in forecast_horizons:
        n_lags = len(horizon_lag_map[h])
        rv_dim = n_lags  # RV每类滞后项1列
        total_dim = rv_dim + exog_dim * n_lags
        print(f"{model_name}（{h}步）：RV维度{rv_dim} + 外生维度{exog_dim * n_lags} = 总维度{total_dim}")
