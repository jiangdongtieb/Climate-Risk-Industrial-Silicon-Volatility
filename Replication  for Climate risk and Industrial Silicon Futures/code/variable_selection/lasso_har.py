import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置（严格遵循论文设定）
# ----------------------
data_path = r"..\..\04_数据\02_处理后数据\回归数据\总回归数据.xlsx"
output_path = r"..\..\results\LASSO-HAR模型结果.xlsx"

# 预测配置（1步=日度，5步=周度，22步=月度，论文2.5节定义）
forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
# 映射：预测步数→需纳入的滞后项类型（论文核心逻辑）
horizon_lag_required = {
    1: ["daily"],  # 1步：仅日度滞后（前1期）
    5: ["daily", "weekly"],  # 5步：日度+周度滞后（前1期+前5期均值）
    22: ["daily", "weekly", "monthly"]  # 22步：日度+周度+月度滞后
}


# 统计损失函数（完全匹配论文2.5.1节公式）
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


# 数据清理函数（处理百分比、字符串、缺失值，贴合论文数据预处理）
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
# 核心：生成外生变量的多层滞后项（按预测步数动态选择）
# ----------------------
def generate_exogenous_lags(exog_data, n_samples):
    """
    生成外生变量的三层滞后项：
    - daily：前1期（日度）
    - weekly：前5期均值（周度）
    - monthly：前22期均值（月度）
    返回：字典{滞后类型: 滞后矩阵}
    """
    n_features = exog_data.shape[1]
    lags = {}

    # 1. 日度滞后（前1期，t-1）
    lags["daily"] = np.roll(exog_data, 1, axis=0)

    # 2. 周度滞后（前5期均值，t-1至t-5）
    weekly = np.zeros_like(exog_data)
    for i in range(5, n_samples):
        weekly[i] = np.mean(exog_data[i - 5:i], axis=0)
    lags["weekly"] = weekly

    # 3. 月度滞后（前22期均值，t-1至t-22）
    monthly = np.zeros_like(exog_data)
    for i in range(22, n_samples):
        monthly[i] = np.mean(exog_data[i - 22:i], axis=0)
    lags["monthly"] = monthly

    # 填充前22期缺失值（用后续有效数据均值，避免NaN）
    for lag_type in lags:
        lags[lag_type][:22] = np.nanmean(lags[lag_type][22:], axis=0)

    return lags


# ----------------------
# 核心：构建每个预测步数的自变量矩阵（按论文逻辑匹配滞后项）
# ----------------------
def build_model_matrix(HAR_lags, exog_lags, horizon):
    """
    按预测步数构建自变量矩阵：
    - HAR_lags：RV的三层滞后项（daily/weekly/monthly）
    - exog_lags：外生变量的三层滞后项
    - horizon：预测步数（1/5/22）
    """
    required_lags = horizon_lag_required[horizon]

    # 1. 拼接RV的滞后项（按需求选择，转为列向量避免维度错误）
    rv_matrix = []
    for lag_type in required_lags:
        rv_matrix.append(HAR_lags[lag_type].reshape(-1, 1))
    rv_matrix = np.hstack(rv_matrix)

    # 2. 拼接外生变量的滞后项（与RV滞后项一一对应）
    exog_matrix = []
    for lag_type in required_lags:
        exog_matrix.append(exog_lags[lag_type])
    exog_matrix = np.hstack(exog_matrix)

    # 3. 合并RV滞后项+外生变量滞后项（论文模型的完整自变量）
    return np.hstack([rv_matrix, exog_matrix])


# ----------------------
# 核心：LASSO变量选择（添加horizon参数，修复未定义报错）
# ----------------------
def lasso_variable_selection(X, y, horizon):
    """
    论文适配版LASSO变量选择：
    输入：X（自变量矩阵）、y（目标变量）、horizon（预测步数，用于匹配滞后项数量）
    输出：筛选后的变量矩阵、变量索引、最优λ、标准化器
    """
    # 标准化（LASSO对尺度敏感，论文强制要求）
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 时间序列交叉验证（5折，避免数据泄露）
    tscv = TimeSeriesSplit(n_splits=5)

    # LASSO交叉验证选λ（100个候选值，论文2.4节实现）
    lasso_cv = LassoCV(
        cv=tscv,
        n_alphas=100,
        random_state=42,
        max_iter=10000,
        tol=1e-3
    )
    lasso_cv.fit(X_scaled, y_scaled)

    # 最优λ与非零系数变量
    best_lambda = lasso_cv.alpha_
    non_zero_idx = np.where(lasso_cv.coef_ != 0)[0]

    # 保底逻辑：最少保留与滞后项数量相等的HAR核心变量（1/2/3个）
    required_lags = horizon_lag_required[horizon]
    n_required_har = len(required_lags)
    if len(non_zero_idx) < n_required_har:
        top_idx = np.argsort(np.abs(lasso_cv.coef_))[::-1][:n_required_har]
        selected_idx = top_idx
    else:
        selected_idx = non_zero_idx

    # 按系数绝对值降序排序
    selected_idx = selected_idx[np.argsort(np.abs(lasso_cv.coef_[selected_idx]))[::-1]]
    X_selected = X[:, selected_idx]
    selected_coef = lasso_cv.coef_[selected_idx]

    # 格式化输出（使用horizon参数匹配预测步数，修复日志打印）
    horizon_name = horizon_names[forecast_horizons.index(horizon)]
    top_coef = np.abs(selected_coef[:5]) if len(selected_coef) >= 5 else np.abs(selected_coef)
    coef_str = ", ".join([f"{val:.4f}" for val in top_coef])
    print(f"LASSO变量筛选（{horizon_name}）："
          f"原始变量数{X.shape[1]} → 保留变量数{len(selected_idx)}")
    print(f"最优惩罚系数λ：{best_lambda:.6f}")
    print(f"保留变量的系数绝对值（前{len(top_coef)}个）：{coef_str}")

    return X_selected, selected_idx, best_lambda, scaler_X, scaler_y


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
n_samples = len(rv)

# 生成RV的三层滞后项（日度/周度/月度，论文2.3节核心结构）
HAR_lags = {
    "daily": np.roll(rv_log, 1),  # 前1天（log(RV_{t-1})）
    "weekly": np.convolve(rv_log, np.ones(5) / 5, mode='same'),  # 前5天均值
    "monthly": np.convolve(rv_log, np.ones(22) / 22, mode='same')  # 前22天均值
}
# 填充RV滞后项的前22天缺失值
for lag_type in HAR_lags:
    HAR_lags[lag_type][:22] = np.nanmean(HAR_lags[lag_type][22:], axis=0)

# 提取外生因子（严格按论文2.2节5组因子）
## 2.1 平均空气污染变量（avgplt-*）
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_exogenous_lags(X_avgplt, n_samples)

## 2.2 加权空气污染变量（weightplt-*）
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_exogenous_lags(X_weightplt, n_samples)

## 2.3 加权气象变量（weather-*）
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_exogenous_lags(X_weather, n_samples)

## 2.4 气候变化关注度（clmfactor前3主成分，论文2.2.4节）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
X_clmfactor_raw = df_clean[clmfactor_cols].values
pca_clm = PCA(n_components=3, random_state=42)
X_clmfactor = pca_clm.fit_transform(X_clmfactor_raw)
exog_lags_clmfactor = generate_exogenous_lags(X_clmfactor, n_samples)

## 2.5 极端天气关注度（weafactor前3主成分，论文2.2.5节）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
pca_wea = PCA(n_components=3, random_state=42)
X_weafactor = pca_wea.fit_transform(X_weafactor_raw)
exog_lags_weafactor = generate_exogenous_lags(X_weafactor, n_samples)

## 2.6 所有外生因子（LASSO-HAR-all模型用，论文模型7）
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_exogenous_lags(X_all, n_samples)

# 构建6个模型的外生因子滞后项字典（便于按步数调用）
models_exog_lags = {
    "LASSO-HAR-avgplt": exog_lags_avgplt,
    "LASSO-HAR-weightplt": exog_lags_weightplt,
    "LASSO-HAR-weather": exog_lags_weather,
    "LASSO-HAR-clmfactor": exog_lags_clmfactor,
    "LASSO-HAR-weafactor": exog_lags_weafactor,
    "LASSO-HAR-all": exog_lags_all
}

# ----------------------
# 3. 滚动窗口预测（核心：传入horizon参数，修复函数调用）
# ----------------------
valid_start = 22  # 去除前22天缺失值（月度滞后项导致）
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)
window_size = max(int(2 / 3 * n_valid), 22)  # 样本内占2/3，最小22天
predictions_dict = {model: {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons} for model in
                    models_exog_lags.keys()}
lasso_info = {model: {h: {} for h in forecast_horizons} for model in models_exog_lags.keys()}  # 存储每个模型+步数的LASSO信息

for h in forecast_horizons:
    required_lags = horizon_lag_required[h]
    print(f"\n=== 开始{h}步超前预测（论文逻辑：{required_lags}滞后项）===")
    # 目标变量：未来h天平均对数RV（直接预测法，论文2.5节定义）
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])

    # 滚动预测
    for i in range(window_size, len(y_target)):
        # 样本内/外索引（转为numpy数组，避免range减法报错）
        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        test_idx_raw = valid_start + i
        pred_date = dates_valid[i + h - 1]  # 预测日期：未来h天的最后一天

        # 样本内数据（按步数构建自变量矩阵）
        X_train_dict = {}
        scaler_dict = {}
        y_train = np.nan_to_num(y_target[train_idx_raw - valid_start], nan=np.nanmean(y_target))

        for model_name, exog_lags in models_exog_lags.items():
            # 构建当前步数的自变量矩阵
            X_raw = build_model_matrix(HAR_lags, exog_lags, h)
            X_train = X_raw[train_idx_raw]

            # LASSO变量选择（首次滚动时执行，传入horizon参数h）
            if len(lasso_info[model_name][h]) == 0:
                # 核心修复：传入h作为horizon参数
                X_lasso, selected_idx, best_lambda, scaler_X, scaler_y = lasso_variable_selection(X_train, y_train, h)
                lasso_info[model_name][h] = {
                    "selected_idx": selected_idx,
                    "best_lambda": best_lambda,
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "original_dim": X_train.shape[1],
                    "selected_dim": len(selected_idx)
                }
            else:
                # 复用首次筛选的结果
                selected_idx = lasso_info[model_name][h]["selected_idx"]
                scaler_X = lasso_info[model_name][h]["scaler_X"]
                X_lasso = X_train[:, selected_idx]

            # 标准化训练数据
            X_train_scaled = scaler_X.fit_transform(X_lasso)
            X_train_dict[model_name] = X_train_scaled
            scaler_dict[model_name] = scaler_X

        # 样本外数据
        X_test_dict = {}
        for model_name, exog_lags in models_exog_lags.items():
            X_raw = build_model_matrix(HAR_lags, exog_lags, h)
            X_test = X_raw[test_idx_raw].reshape(1, -1)
            selected_idx = lasso_info[model_name][h]["selected_idx"]
            X_test_lasso = X_test[:, selected_idx]
            scaler_X = scaler_dict[model_name]
            X_test_scaled = scaler_X.transform(X_test_lasso)
            X_test_dict[model_name] = X_test_scaled

        # 真实值（反对数化还原）
        true_rv = np.expm1(y_target[i])

        # 训练并预测
        for model_name, X_train_scaled in X_train_dict.items():
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred_log = model.predict(X_test_dict[model_name])[0]
            y_pred_rv = np.expm1(y_pred_log)

            # 记录结果
            predictions_dict[model_name][h]['dates'].append(pred_date)
            predictions_dict[model_name][h]['pred'].append(y_pred_rv)
            predictions_dict[model_name][h]['true'].append(true_rv)

# ----------------------
# 4. 统计指标与结果保存（MSE保留8位小数，贴合论文表格）
# ----------------------
stats_results = []
for model_name in models_exog_lags.keys():
    for h in forecast_horizons:
        y_true = np.array(predictions_dict[model_name][h]['true'])
        y_pred = np.array(predictions_dict[model_name][h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "有效样本数": valid_len,
            "原始变量数": lasso_info[model_name][h]["original_dim"],
            "保留变量数": lasso_info[model_name][h]["selected_dim"],
            "最优λ": round(lasso_info[model_name][h]["best_lambda"], 6),
            "MSE": round(mse, 8),  # 核心：MSE保留8位小数
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6)
        })

stats_df = pd.DataFrame(stats_results)

# LASSO变量选择信息汇总（便于论文报告）
lasso_summary = []
for model_name in models_exog_lags.keys():
    for h in forecast_horizons:
        info = lasso_info[model_name][h]
        lasso_summary.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "原始变量数": info["original_dim"],
            "保留变量数": info["selected_dim"],
            "变量筛选比例": round(info["selected_dim"] / info["original_dim"], 4),
            "最优惩罚系数λ": round(info["best_lambda"], 6),
            "纳入滞后项类型": ", ".join(horizon_lag_required[h])
        })
lasso_summary_df = pd.DataFrame(lasso_summary)

# 保存结果（多工作表，符合论文输出格式）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)
    lasso_summary_df.to_excel(writer, sheet_name="LASSO筛选信息", index=False)
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        step_data = None
        for model_name in models_exog_lags.keys():
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
print("\n=== LASSO变量选择结果汇总 ===")
print(lasso_summary_df[["模型名称", "预测步数", "原始变量数", "保留变量数", "纳入滞后项类型"]].head(12))
print(f"\n📊 统计指标汇总（前10行）：")
print(stats_df[["模型名称", "预测步数", "保留变量数", "MSE", "QLIKE"]].head(10))
