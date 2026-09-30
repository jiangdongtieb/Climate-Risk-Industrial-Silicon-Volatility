import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import hashlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置
# ----------------------
data_path = r"C:\Users\35632\Desktop\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\LASSO-HAR-all_结果111.xlsx"
fig_save_path = r"C:\Users\35632\Desktop\LASSO-HAR-all_可视化结果111"

# 创建可视化目录
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# 预测配置
forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]
horizon_lag_required = {
    1: ["daily"],
    5: ["daily", "weekly"],
    22: ["daily", "weekly", "monthly"]
}

# 所有可能的滞后类型（用于统一变量列表）
ALL_LAG_TYPES = ["daily", "weekly", "monthly"]


# ----------------------
# 工具函数
# ----------------------
def calculate_loss(y_true, y_pred):
    """计算损失函数"""
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
    """数据清理函数"""
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
    """主成分分析，只返回数据"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_scaled)

    # 计算载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'{name}_PC{i + 1}' for i in range(n_components)],
        index=cols
    )

    # 变量贡献度
    var_contribution = np.abs(loadings).sum(axis=1)
    var_contribution = var_contribution / var_contribution.sum() * 100
    loadings_df['总贡献度(%)'] = var_contribution.round(2)

    # 贡献率
    exp_var = pca.explained_variance_ratio_ * 100
    exp_var_df = pd.DataFrame({
        '主成分': [f'{name}_PC{i + 1}' for i in range(n_components)],
        '方差贡献率(%)': exp_var.round(2),
        '累计贡献率(%)': np.cumsum(exp_var).round(2)
    })

    return data_pca, loadings_df, exp_var_df, scaler


def generate_lags(data, n_samples):
    """统一生成三层滞后项"""
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    lags = {}

    # 日度滞后（前1期）
    lags["daily"] = np.roll(data, 1, axis=0)

    # 周度滞后（前5期均值）
    weekly = np.zeros_like(data)
    for i in range(5, n_samples):
        weekly[i] = np.mean(data[i - 5:i], axis=0)
    lags["weekly"] = weekly

    # 月度滞后（前22期均值）
    monthly = np.zeros_like(data)
    for i in range(22, n_samples):
        monthly[i] = np.mean(data[i - 22:i], axis=0)
    lags["monthly"] = monthly

    # 填充前22期缺失值
    for lag_type in lags:
        lags[lag_type][:22] = np.nanmean(lags[lag_type][22:], axis=0)

    return lags


def build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, horizon):
    """构建自变量矩阵"""
    required_lags = horizon_lag_required[horizon]

    # 拼接RV滞后项
    rv_matrix = []
    rv_var_names = []
    for lag_type in required_lags:
        rv_col = rv_lags[lag_type]
        rv_matrix.append(rv_col)
        rv_var_names.append(f'RV_{lag_type}')
    rv_matrix = np.hstack(rv_matrix) if len(rv_matrix) > 1 else rv_matrix[0]

    # 拼接外生变量滞后项
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

    # 合并RV+外生变量
    X_matrix = np.hstack([rv_matrix, exog_matrix])
    full_var_names = rv_var_names + exog_var_names_full

    return X_matrix, full_var_names


def get_full_variable_list(exog_var_names):
    """
    获取完整的变量列表（包含所有滞后类型）
    用于统一不同步长的变量显示
    """
    full_var_list = []

    # 添加RV变量（所有滞后类型）
    for lag_type in ALL_LAG_TYPES:
        full_var_list.append(f'RV_{lag_type}')

    # 添加外生变量（所有滞后类型）
    for var in exog_var_names:
        for lag_type in ALL_LAG_TYPES:
            full_var_list.append(f'{var}_{lag_type}')

    return full_var_list


def create_selection_matrix_for_all_lags(selection_matrix_original, var_names_original, all_var_names):
    """
    将选择矩阵扩展到完整的变量列表
    未使用的变量位置填充NaN（显示为灰色）
    """
    # 创建映射：原始变量名 -> 在原始矩阵中的列索引
    var_to_idx = {var: idx for idx, var in enumerate(var_names_original)}

    # 创建完整的矩阵
    n_timepoints = selection_matrix_original.shape[0]
    full_matrix = np.full((n_timepoints, len(all_var_names)), np.nan)

    # 填充已有数据
    for col_idx, var_name in enumerate(all_var_names):
        if var_name in var_to_idx:
            original_idx = var_to_idx[var_name]
            full_matrix[:, col_idx] = selection_matrix_original[:, original_idx]

    return full_matrix


def export_selection_to_excel(writer, selection_matrices, full_var_names, horizon_names, forecast_horizons):
    """
    将变量选择矩阵导出到Excel，每个步长一个工作表
    使用条件格式进行可视化
    """
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]

        if len(selection_matrices[h]['matrix']) == 0:
            continue

        # 获取数据
        selection_matrix = np.array(selection_matrices[h]['matrix'])
        dates = selection_matrices[h]['dates']
        var_names_original = selection_matrices[h]['var_names']

        # 扩展到完整变量列表
        selection_matrix_full = create_selection_matrix_for_all_lags(
            selection_matrix, var_names_original, full_var_names
        )

        # 创建DataFrame
        df_selection = pd.DataFrame(
            selection_matrix_full.T,  # 转置，让变量作为行，时间作为列
            index=full_var_names,
            columns=[d.strftime('%Y-%m-%d') for d in dates]
        )

        # 添加统计信息列
        df_selection['选中次数'] = df_selection.apply(
            lambda row: row[row == 1].count(), axis=1
        )
        df_selection['选中频率(%)'] = (df_selection['选中次数'] / len(dates) * 100).round(2)

        # 按选中频率排序
        df_selection_sorted = df_selection.sort_values('选中频率(%)', ascending=False)

        # 写入Excel
        sheet_name = f'{horizon_name}_变量选择'
        df_selection_sorted.to_excel(writer, sheet_name=sheet_name)

        # 获取工作表并添加格式
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # 定义颜色填充
        green_fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")  # 选中（绿色）
        gray_fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid")  # 未选中（灰色）
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")  # 不适用（白色）

        # 为每个单元格添加颜色
        print(f"  正在为 {sheet_name} 添加颜色格式...")
        for row_idx, var_name in enumerate(df_selection_sorted.index, start=2):
            for col_idx, date in enumerate(df_selection_sorted.columns[:-2], start=2):  # 跳过统计列
                cell = worksheet.cell(row=row_idx, column=col_idx)
                value = cell.value

                if pd.isna(value):
                    cell.fill = white_fill
                elif value == 1:
                    cell.fill = green_fill
                    cell.font = Font(color="000000", bold=True)
                elif value == 0:
                    cell.fill = gray_fill
                else:
                    cell.fill = white_fill

        # 添加边框
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        for row in worksheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    cell.border = thin_border

        # 设置列宽
        worksheet.column_dimensions['A'].width = 35  # 变量名列宽
        for col_idx in range(2, len(df_selection_sorted.columns) - 1):
            col_letter = openpyxl.utils.get_column_letter(col_idx)
            worksheet.column_dimensions[col_letter].width = 12

        # 设置表头样式
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        for cell in worksheet[1]:
            cell.font = Font(bold=True, size=11, color="FFFFFF")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.fill = header_fill

        # 设置统计列样式
        last_col = len(df_selection_sorted.columns)
        for row_idx in range(2, len(df_selection_sorted.index) + 2):
            # 选中次数列
            cell_count = worksheet.cell(row=row_idx, column=last_col - 1)
            cell_count.alignment = Alignment(horizontal='center', vertical='center')
            # 选中频率列
            cell_freq = worksheet.cell(row=row_idx, column=last_col)
            cell_freq.alignment = Alignment(horizontal='center', vertical='center')
            cell_freq.number_format = '0.00"%"'

        print(f"  已导出 {sheet_name} 到Excel（共{len(df_selection_sorted)}个变量，{len(dates)}个时间点）")


def lasso_variable_selection_enhanced(X, y, horizon, var_names):
    """LASSO变量选择"""
    # 标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
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

    # 保底逻辑：最少保留HAR核心变量
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
    selected_var_names = [var_names[i] for i in selected_idx]

    # 格式化输出
    horizon_name = horizon_names[forecast_horizons.index(horizon)]
    print(f"LASSO变量筛选（{horizon_name}）：原始变量数{X.shape[1]} → 保留变量数{len(selected_idx)}")
    print(f"最优惩罚系数λ：{best_lambda:.6f}")

    return X_selected, selected_idx, selected_var_names, best_lambda, scaler_X, scaler_y, lasso_cv.coef_


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
rv_log = np.log1p(rv)
n_samples = len(rv)

# 生成RV的三层滞后项
rv_lags = generate_lags(rv_log, n_samples)

# 提取外生因子
# 平均空气污染
avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_lags(X_avgplt, n_samples)

# 加权空气污染
weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_lags(X_weightplt, n_samples)

# 加权气象变量
weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
print(f"Weather变量数量：{len(weather_cols)}")
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_lags(X_weather, n_samples)

# 气候变化关注度（前3主成分）
clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
print(f"CLMFactor变量数量：{len(clmfactor_cols)}")
X_clmfactor_raw = df_clean[clmfactor_cols].values
X_clmfactor, clm_loadings, clm_exp_var, clm_scaler = pca_analysis_enhanced(
    X_clmfactor_raw, clmfactor_cols, 3, 'CLMFactor'
)
exog_lags_clmfactor = generate_lags(X_clmfactor, n_samples)
clmfactor_pc_names = [f'CLMFactor_PC{i + 1}' for i in range(3)]

# 极端天气关注度（前3主成分）
weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
X_weafactor, wea_loadings, wea_exp_var, wea_scaler = pca_analysis_enhanced(
    X_weafactor_raw, weafactor_cols, 3, 'WEAFactor'
)
exog_lags_weafactor = generate_lags(X_weafactor, n_samples)
weafactor_pc_names = [f'WEAFactor_PC{i + 1}' for i in range(3)]

# 所有外生因子
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_lags(X_all, n_samples)
all_exog_names = avgplt_cols + weightplt_cols + weather_cols + clmfactor_pc_names + weafactor_pc_names

# 只保留LASSO-HAR-all模型
model_name = "LASSO-HAR-all"
exog_lags = exog_lags_all
exog_var_names = all_exog_names

# 获取完整的变量列表（包含所有滞后类型）
full_var_names = get_full_variable_list(exog_var_names)
print(f"完整变量列表数量：{len(full_var_names)}")

# 去除前22天缺失值
valid_start = 22
dates_valid = df_clean.index[valid_start:]
n_valid = len(dates_valid)

# ----------------------
# 3. 滚动窗口预测
# ----------------------
window_size = max(int(2 / 3 * n_valid), 22)
predictions = {h: {'dates': [], 'pred': [], 'true': []} for h in forecast_horizons}
lasso_details = {h: [] for h in forecast_horizons}
selection_matrices = {h: {'matrix': [], 'var_names': None, 'dates': []} for h in forecast_horizons}
var_selection_summary = []

print("\n开始滚动窗口预测...")
for h in forecast_horizons:
    required_lags = horizon_lag_required[h]
    horizon_name = horizon_names[forecast_horizons.index(h)]
    print(f"\n=== 开始{h}步超前预测（{horizon_name}：{required_lags}滞后项）===")

    # 目标变量：未来h天平均对数RV
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])

    # 滚动预测
    for i in range(window_size, len(y_target)):
        if i % 50 == 0:
            print(f"  处理进度：{i - window_size + 1}/{len(y_target) - window_size}")

        # 样本索引
        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        test_idx_raw = valid_start + i
        pred_date = dates_valid[i + h - 1]
        y_train = np.nan_to_num(y_target[i - window_size: i], nan=np.nanmean(y_target))
        true_rv = np.expm1(y_target[i])

        # 构建矩阵+变量名
        X_raw, var_names_original = build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, h)

        # 样本内/外数据
        X_train = X_raw[train_idx_raw]
        X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
        X_test = X_raw[test_idx_raw].reshape(1, -1)
        X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))

        # LASSO变量选择
        X_lasso, selected_idx, selected_var_names, best_lambda, scaler_X, scaler_y, all_coef = lasso_variable_selection_enhanced(
            X_train, y_train, h, var_names_original
        )

        # 记录LASSO信息
        lasso_details[h].append({
            'date': pred_date,
            'selected_idx': selected_idx,
            'selected_var_names': selected_var_names,
            'best_lambda': best_lambda,
            'all_coef': all_coef,
            'original_dim': X_train.shape[1],
            'selected_dim': len(selected_idx)
        })

        # 存储选择矩阵（用于热力图）
        if selection_matrices[h]['var_names'] is None:
            selection_matrices[h]['var_names'] = var_names_original
        selection_matrices[h]['dates'].append(pred_date)

        # 创建选择向量（1表示选中，0表示未选中）
        selection_vector = np.zeros(len(var_names_original))
        selection_vector[selected_idx] = 1
        selection_matrices[h]['matrix'].append(selection_vector)

        # 标准化+预测
        X_train_scaled = scaler_X.fit_transform(X_lasso)
        X_test_lasso = X_test[:, selected_idx]
        X_test_scaled = scaler_X.transform(X_test_lasso)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)[0]
        y_pred_rv = np.expm1(y_pred_log)

        # 记录预测结果
        predictions[h]['dates'].append(pred_date)
        predictions[h]['pred'].append(y_pred_rv)
        predictions[h]['true'].append(true_rv)

# ----------------------
# 4. 导出变量选择到Excel（核心功能）
# ----------------------
print("\n=== 导出变量选择结果到Excel ===")

# 使用ExcelWriter并启用openpyxl引擎
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 统计指标
    stats_results = []
    for h in forecast_horizons:
        y_true = np.array(predictions[h]['true'])
        y_pred = np.array(predictions[h]['pred'])
        valid_len = len(y_true)
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)

        first_lasso = lasso_details[h][0] if len(lasso_details[h]) > 0 else {}

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
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)

    # 变量选择汇总（频率统计）
    for h in forecast_horizons:
        horizon_name = horizon_names[forecast_horizons.index(h)]
        all_vars = set()
        selected_counts = {}
        total_records = len(lasso_details[h])

        for record in lasso_details[h]:
            all_vars.update(record['selected_var_names'])
            for var in record['selected_var_names']:
                selected_counts[var] = selected_counts.get(var, 0) + 1

        for var in all_vars:
            freq = selected_counts.get(var, 0) / total_records * 100 if total_records > 0 else 0
            var_selection_summary.append({
                "模型名称": model_name,
                "预测步数": horizon_name,
                "变量名称": var,
                "选中次数": selected_counts.get(var, 0),
                "总预测次数": total_records,
                "选中频率(%)": round(freq, 2)
            })

    var_selection_df = pd.DataFrame(var_selection_summary)
    var_selection_df.to_excel(writer, sheet_name="LASSO变量选择频率", index=False)

    # PCA结果
    clm_loadings.to_excel(writer, sheet_name="CLMFactor_PCA载荷", index=True)
    clm_exp_var.to_excel(writer, sheet_name="CLMFactor_PCA贡献率", index=False)
    wea_loadings.to_excel(writer, sheet_name="WEAFactor_PCA载荷", index=True)
    wea_exp_var.to_excel(writer, sheet_name="WEAFactor_PCA贡献率", index=False)

    # 预测值
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]
        result_df = pd.DataFrame({
            "日期": predictions[h]['dates'],
            f"{model_name}_预测RV": [round(val, 6) for val in predictions[h]['pred']],
            f"{model_name}_真实RV": [round(val, 6) for val in predictions[h]['true']]
        })
        result_df.to_excel(writer, sheet_name=f"{horizon_name}预测值", index=False)

    # 导出变量选择矩阵（带颜色格式的表格）
    print("\n正在生成带格式的变量选择表格...")
    export_selection_to_excel(writer, selection_matrices, full_var_names, horizon_names, forecast_horizons)

print(f"\n✅ 所有结果已保存至：{output_path}")
print(f"\n📊 Excel文件包含以下工作表：")
print("   - 统计指标汇总")
print("   - LASSO变量选择频率")
print("   - 各步长的详细变量选择矩阵（带颜色标注）")
print("   - PCA分析结果")
print("   - 预测值")
print("\n💡 提示：")
print("   1. 在Excel中，您可以自由调整字体大小、颜色、行高等")
print("   2. 变量选择矩阵中：绿色=选中，灰色=未选中，白色=不适用")
print("   3. 可以使用Excel的条件格式功能进一步美化")

# ----------------------
# 5. 输出汇总
# ----------------------
print("\n=== LASSO变量选择结果汇总 ===")
print(var_selection_df.head(12))
print(f"\n📊 统计指标汇总：")
print(stats_df)

print("\n✅ 分析完成！")