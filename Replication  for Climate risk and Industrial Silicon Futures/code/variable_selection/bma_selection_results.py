import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os
from itertools import combinations
from scipy import stats
import time
import hashlib
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置
# ----------------------
data_path = r"C:\Users\35632\Desktop\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\BMA-HAR-all_作图版1.xlsx"
fig_save_path = r"C:\Users\35632\Desktop\BMA-HAR-all_30-5000-0.15作图"

forecast_horizons = [1, 5, 22]
horizon_names = ["1步超前", "5步超前", "22步超前"]

horizon_lag_map = {
    1: ["daily"],
    5: ["daily", "weekly"],
    22: ["daily", "weekly", "monthly"]
}

# 所有可能的滞后类型
ALL_LAG_TYPES = ["daily", "weekly", "monthly"]

BMA_MAX_VARS = 30
BMA_MIN_PROB = 0.01

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)


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


def pca_analysis(data, cols, n_components, name):
    """PCA分析，只返回数据，不生成图"""
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)
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
    return data_pca, loadings_df, exp_var_df


def generate_all_lags(data, n_samples):
    n_features = data.shape[1] if len(data.shape) > 1 else 1
    if n_features == 1:
        data = data.reshape(-1, 1)
    daily = np.roll(data, 1, axis=0)
    weekly = np.zeros_like(data)
    for i in range(5, n_samples):
        weekly[i] = np.mean(data[i - 5:i], axis=0)
    monthly = np.zeros_like(data)
    for i in range(22, n_samples):
        monthly[i] = np.mean(data[i - 22:i], axis=0)
    daily[:22] = np.nanmean(daily[22:], axis=0)
    weekly[:22] = np.nanmean(weekly[22:], axis=0)
    monthly[:22] = np.nanmean(monthly[22:], axis=0)
    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def build_model_matrix_and_names(rv_lags, exog_lags, exog_var_names, horizon, include_rv=True):
    required_lags = horizon_lag_map[horizon]
    all_matrices = []
    all_var_names = []
    if include_rv:
        for lag_type in required_lags:
            rv_col = rv_lags[lag_type].reshape(-1, 1)
            all_matrices.append(rv_col)
            all_var_names.append(f'RV_{lag_type}')
    if exog_lags is not None:
        for lag_type in required_lags:
            exog_col = exog_lags[lag_type]
            all_matrices.append(exog_col)
            exog_var_names_lag = [f'{var}_{lag_type}' for var in exog_var_names]
            all_var_names.extend(exog_var_names_lag)
    if len(all_matrices) == 0:
        return None, []
    X_matrix = np.hstack(all_matrices)
    assert X_matrix.shape[1] == len(all_var_names)
    return X_matrix, all_var_names


def get_full_variable_list(exog_var_names):
    """获取完整的变量列表（包含所有滞后类型）"""
    full_var_list = []

    # 添加RV变量（所有滞后类型）
    for lag_type in ALL_LAG_TYPES:
        full_var_list.append(f'RV_{lag_type}')

    # 添加外生变量（所有滞后类型）
    for var in exog_var_names:
        for lag_type in ALL_LAG_TYPES:
            full_var_list.append(f'{var}_{lag_type}')

    return full_var_list


def create_prob_matrix_for_all_lags(prob_matrix_original, var_names_original, all_var_names):
    """将概率矩阵扩展到完整的变量列表"""
    var_to_idx = {var: idx for idx, var in enumerate(var_names_original)}

    n_timepoints = prob_matrix_original.shape[0]
    full_matrix = np.full((n_timepoints, len(all_var_names)), np.nan)

    for col_idx, var_name in enumerate(all_var_names):
        if var_name in var_to_idx:
            original_idx = var_to_idx[var_name]
            full_matrix[:, col_idx] = prob_matrix_original[:, original_idx]

    return full_matrix


def export_bma_probability_to_excel(writer, prob_matrices, full_var_names, horizon_names, forecast_horizons):
    """
    将BMA概率矩阵导出到Excel，每个步长一个工作表
    使用颜色渐变表示概率大小
    """
    for h_idx, h in enumerate(forecast_horizons):
        horizon_name = horizon_names[h_idx]

        if len(prob_matrices[h]['matrix']) == 0:
            continue

        # 获取数据
        prob_matrix_original = np.array(prob_matrices[h]['matrix'])
        var_names_original = prob_matrices[h]['var_names']
        dates = prob_matrices[h]['dates']

        # 扩展到完整变量列表
        prob_matrix_full = create_prob_matrix_for_all_lags(
            prob_matrix_original, var_names_original, full_var_names
        )

        # 创建DataFrame（转置，让变量作为行，时间作为列）
        df_prob = pd.DataFrame(
            prob_matrix_full.T,  # 转置
            index=full_var_names,
            columns=[d.strftime('%Y-%m-%d') for d in dates]
        )

        # 添加统计信息列
        df_prob['平均概率'] = df_prob.mean(axis=1).round(4)
        df_prob['最大概率'] = df_prob.max(axis=1).round(4)
        df_prob['最小概率'] = df_prob.min(axis=1).round(4)
        df_prob['概率标准差'] = df_prob.std(axis=1).round(4)

        # 按平均概率排序
        df_prob_sorted = df_prob.sort_values('平均概率', ascending=False)

        # 写入Excel
        sheet_name = f'{horizon_name}_BMA概率'
        df_prob_sorted.to_excel(writer, sheet_name=sheet_name)

        # 获取工作表并添加格式
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # 定义颜色填充函数（基于概率值）- 修复类型错误
        def get_color_for_prob(prob):
            # 确保prob是数值类型
            try:
                prob_value = float(prob) if prob is not None else None
            except (ValueError, TypeError):
                prob_value = None

            if prob_value is None or pd.isna(prob_value):
                return PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
            # 绿色渐变：概率越高颜色越深
            if prob_value >= 0.8:
                return PatternFill(start_color="006400", end_color="006400", fill_type="solid")  # 深绿
            elif prob_value >= 0.6:
                return PatternFill(start_color="228B22", end_color="228B22", fill_type="solid")  # 森林绿
            elif prob_value >= 0.4:
                return PatternFill(start_color="32CD32", end_color="32CD32", fill_type="solid")  # 石灰绿
            elif prob_value >= 0.2:
                return PatternFill(start_color="7CFC00", end_color="7CFC00", fill_type="solid")  # 草绿
            elif prob_value >= 0.05:
                return PatternFill(start_color="ADFF2F", end_color="ADFF2F", fill_type="solid")  # 黄绿
            else:
                return PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid")  # 浅灰

        print(f"  正在为 {sheet_name} 添加颜色格式...")

        # 为每个单元格添加颜色
        for row_idx, var_name in enumerate(df_prob_sorted.index, start=2):
            for col_idx, date in enumerate(df_prob_sorted.columns[:-4], start=2):  # 跳过统计列
                cell = worksheet.cell(row=row_idx, column=col_idx)
                prob_value = cell.value

                # 跳过非数值单元格（如NaN）
                if prob_value is not None and prob_value != '':
                    try:
                        # 尝试转换为数值
                        numeric_value = float(prob_value)
                        cell.fill = get_color_for_prob(numeric_value)
                        # 设置数字格式为百分比
                        cell.number_format = '0.00%'
                    except (ValueError, TypeError):
                        # 如果无法转换，保持原样
                        pass

        # 为统计列添加格式
        stats_cols = ['平均概率', '最大概率', '最小概率', '概率标准差']
        for col_idx, stat_name in enumerate(stats_cols, start=len(df_prob_sorted.columns) - 3):
            for row_idx in range(2, len(df_prob_sorted.index) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx + 1)
                if stat_name in ['平均概率', '最大概率', '最小概率']:
                    try:
                        cell_value = float(cell.value) if cell.value is not None else None
                        if cell_value is not None:
                            cell.number_format = '0.00%'
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        cell_value = float(cell.value) if cell.value is not None else None
                        if cell_value is not None:
                            cell.number_format = '0.0000'
                    except (ValueError, TypeError):
                        pass
                cell.alignment = Alignment(horizontal='center', vertical='center')

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
        worksheet.column_dimensions['A'].width = 40  # 变量名列宽
        for col_idx in range(2, len(df_prob_sorted.columns) - 3):
            col_letter = get_column_letter(col_idx)
            worksheet.column_dimensions[col_letter].width = 12

        # 设置表头样式
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        for cell in worksheet[1]:
            cell.font = Font(bold=True, size=11, color="FFFFFF")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.fill = header_fill

        print(f"  已导出 {sheet_name} 到Excel（共{len(df_prob_sorted)}个变量，{len(dates)}个时间点）")


def plot_bma_probability_heatmap(prob_matrix, var_names, dates, title, filename):
    """
    绘制BMA概率热力图（彩色版本，无颜色条）
    prob_matrix: 二维数组，行为时间点，列为变量，值为包含概率(0-1)
    """
    n_vars = len(var_names)
    n_dates = len(dates)

    # 动态调整图形大小
    fig_width = max(20, n_dates / 8)
    fig_height = max(16, n_vars / 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 为每个变量创建独特的颜色映射
    unique_vars = var_names
    var_colors = {}
    for var in unique_vars:
        var_colors[var] = get_variable_color(var)

    # 创建颜色数组
    colored_matrix = np.zeros((prob_matrix.shape[1], prob_matrix.shape[0], 4))

    for i, var in enumerate(unique_vars):
        color = var_colors[var]
        for j in range(prob_matrix.shape[0]):
            prob_value = prob_matrix[j, i]
            if not np.isnan(prob_value):
                # 透明度完全由概率值决定
                colored_matrix[i, j] = (*color, prob_value)
            else:
                # 缺失值：浅灰色
                colored_matrix[i, j] = (0.95, 0.95, 0.95, 1.0)

    ax.imshow(colored_matrix, aspect='auto', interpolation='nearest')

    # 添加周期性垂直背景色带（每3个月交替）
    if n_dates > 0:
        months = pd.Series(dates).dt.to_period('M')
        unique_months = months.unique()
        for i, month in enumerate(unique_months):
            month_indices = np.where(months == month)[0]
            if len(month_indices) > 0:
                start_idx = month_indices[0]
                end_idx = month_indices[-1] + 1
                if (i // 3) % 2 == 0:
                    ax.axvspan(start_idx - 0.5, end_idx - 0.5, facecolor='#f0f0f0', alpha=0.2, zorder=-1)

    # 设置坐标轴
    ax.set_yticks(np.arange(len(unique_vars)))
    y_fontsize = max(10, min(16, 300 / max(1, n_vars) * 2))
    ax.set_yticklabels(unique_vars, fontsize=y_fontsize, fontweight='bold')

    step = max(1, n_dates // 30)
    if step < 2:
        step = 2
    xtick_positions = np.arange(0, n_dates, step)
    xtick_labels = [dates[i].strftime('%Y-%m') for i in xtick_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=12)

    ax.set_xlabel('预测时点', fontsize=16, fontweight='bold')
    ax.set_ylabel('预测变量', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')

    # 添加图例
    legend_elements = []
    legend_elements.append(
        mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='不适用（该步长不使用此滞后项）'))
    legend_elements.append(mpatches.Patch(facecolor='white', alpha=0.3, edgecolor='gray', label='包含概率较低'))
    legend_elements.append(mpatches.Patch(facecolor='white', alpha=0.8, edgecolor='gray', label='包含概率较高'))

    # 添加代表性变量的图例（最多15个）
    sample_vars = unique_vars[:15]
    for var in sample_vars:
        legend_elements.append(mpatches.Patch(facecolor=var_colors[var], alpha=0.8, label=var))

    if len(unique_vars) > 15:
        legend_elements.append(mpatches.Patch(facecolor='lightgray', label=f'... (共{len(unique_vars)}个变量)'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True,
              shadow=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def get_variable_color(var_name):
    """为每个变量分配独特的颜色"""
    hash_obj = hashlib.md5(var_name.encode())
    hash_hex = hash_obj.hexdigest()

    r = int(hash_hex[0:2], 16) / 255.0
    g = int(hash_hex[2:4], 16) / 255.0
    b = int(hash_hex[4:6], 16) / 255.0

    brightness = (r + g + b) / 3
    if brightness < 0.3:
        r = min(1.0, r + 0.3)
        g = min(1.0, g + 0.3)
        b = min(1.0, b + 0.3)
    elif brightness > 0.8:
        r = max(0.0, r - 0.2)
        g = max(0.0, g - 0.2)
        b = max(0.0, b - 0.2)

    return (r, g, b)


def calculate_bma_weights_and_coef(X_train, y_train, var_names, max_vars=BMA_MAX_VARS):
    """计算BMA权重并返回每个变量的平均系数"""
    n_samples = len(y_train)
    n_vars = X_train.shape[1]
    if n_vars == 0:
        return np.array([]), {}, np.array([])

    X_scaled = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
    y_scaled = y_train - np.mean(y_train)

    candidate_models = [[]]
    for k in range(1, min(n_vars, max_vars) + 1):
        combs = combinations(range(n_vars), k)
        candidate_models.extend([list(comb) for comb in combs])
        if len(candidate_models) > 5000:
            break

    bics = []
    coefs_list = []

    for var_indices in candidate_models:
        if len(var_indices) == 0:
            rss = np.sum((y_scaled - np.mean(y_scaled)) ** 2)
            k = 0
            coefs = np.zeros(n_vars)
        else:
            X_sub = X_scaled[:, var_indices]
            model = LinearRegression()
            model.fit(X_sub, y_scaled)
            y_pred = model.predict(X_sub)
            rss = np.sum((y_scaled - y_pred) ** 2)
            k = len(var_indices) + 1
            # 构建完整系数向量
            coefs = np.zeros(n_vars)
            for idx, coef in zip(var_indices, model.coef_):
                coefs[idx] = coef
        bic = n_samples * np.log(rss / n_samples + 1e-10) + k * np.log(n_samples) * 0.5
        bics.append(bic)
        coefs_list.append(coefs)

    bics = np.array(bics)
    min_bic = np.min(bics)
    weights = np.exp(-0.5 * (bics - min_bic))
    weights /= np.sum(weights) + 1e-10

    # 计算后验包含概率和平均系数
    inclusion_probs = np.zeros(n_vars)
    avg_coefs = np.zeros(n_vars)

    for idxs, w in zip(candidate_models, weights):
        for idx in idxs:
            inclusion_probs[idx] += w
        avg_coefs += coefs_list[idx] * w

    var_probs = {var_names[i]: float(inclusion_probs[i]) for i in range(n_vars)}
    var_coefs = {var_names[i]: float(avg_coefs[i]) for i in range(n_vars)}

    return inclusion_probs, var_probs, avg_coefs


# ----------------------
# 2. 数据加载与预处理
# ----------------------
print("正在加载数据...")
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df_clean = clean_data(df)

rv = df_clean['rv'].values
n_samples = len(rv)
rv_log = np.log1p(rv)

rv_lags = generate_all_lags(rv_log, n_samples)

avgplt_cols = [col for col in df_clean.columns if col.startswith('avgplt-')]
X_avgplt = df_clean[avgplt_cols].values
exog_lags_avgplt = generate_all_lags(X_avgplt, n_samples)

weightplt_cols = [col for col in df_clean.columns if col.startswith('weightplt-')]
X_weightplt = df_clean[weightplt_cols].values
exog_lags_weightplt = generate_all_lags(X_weightplt, n_samples)

weather_cols = [col for col in df_clean.columns if col.startswith('weather-')]
weather_cols = [col for col in weather_cols if 'pm' not in col.lower() and 'aqi' not in col.lower()]
print(f"Weather变量数量：{len(weather_cols)}")
X_weather = df_clean[weather_cols].values
exog_lags_weather = generate_all_lags(X_weather, n_samples)

clmfactor_cols = [col for col in df_clean.columns if col.startswith('clmfactor-')]
print(f"CLMFactor变量数量：{len(clmfactor_cols)}")
X_clmfactor_raw = df_clean[clmfactor_cols].values
X_clmfactor, clm_loadings, clm_exp_var = pca_analysis(X_clmfactor_raw, clmfactor_cols, 3, 'CLMFactor')
exog_lags_clmfactor = generate_all_lags(X_clmfactor, n_samples)
clmfactor_pc_names = [f'CLMFactor_PC{i + 1}' for i in range(3)]

weafactor_cols = [col for col in df_clean.columns if col.startswith('weafactor-')]
X_weafactor_raw = df_clean[weafactor_cols].values
X_weafactor, wea_loadings, wea_exp_var = pca_analysis(X_weafactor_raw, weafactor_cols, 3, 'WEAFactor')
exog_lags_weafactor = generate_all_lags(X_weafactor, n_samples)
weafactor_pc_names = [f'WEAFactor_PC{i + 1}' for i in range(3)]

# 所有外生因子
X_all = np.hstack([X_avgplt, X_weightplt, X_weather, X_clmfactor, X_weafactor])
exog_lags_all = generate_all_lags(X_all, n_samples)
all_exog_names = avgplt_cols + weightplt_cols + weather_cols + clmfactor_pc_names + weafactor_pc_names

# 只保留BMA-HAR-all模型
model_name = "BMA-HAR-all"
rv_lags_model = rv_lags
exog_lags_model = exog_lags_all
exog_var_names_model = all_exog_names

# 获取完整的变量列表（包含所有滞后类型）
full_var_names = get_full_variable_list(exog_var_names_model)
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
bma_details = {h: [] for h in forecast_horizons}
prob_matrices = {h: {'matrix': [], 'var_names': None, 'dates': []} for h in forecast_horizons}
coef_details = {h: [] for h in forecast_horizons}

print("\n开始BMA滚动窗口预测...")
start_time = time.time()

for h in forecast_horizons:
    print(f"\n正在执行{h}步超前BMA预测...")
    y_target = np.zeros(n_valid - h + 1)
    for i in range(len(y_target)):
        y_target[i] = np.mean(rv_log[valid_start + i: valid_start + i + h])
    total_windows = len(y_target) - window_size

    for i in range(window_size, len(y_target)):
        if i % 50 == 0:
            print(f"  进度：{i - window_size + 1}/{total_windows}")

        train_start = valid_start + i - window_size
        train_end = valid_start + i
        train_idx_raw = np.arange(train_start, train_end)
        pred_date = dates_valid[i + h - 1]
        y_train = np.nan_to_num(y_target[i - window_size: i], nan=np.nanmean(y_target))
        true_rv = np.expm1(y_target[i])

        # 构建矩阵+变量名
        X_full, var_names_original = build_model_matrix_and_names(
            rv_lags_model, exog_lags_model, exog_var_names_model, h, include_rv=True
        )
        X_train = np.nan_to_num(X_full[train_idx_raw], nan=np.nanmean(X_full))
        X_test = X_full[valid_start + i].reshape(1, -1)
        X_test = np.nan_to_num(X_test)

        try:
            inclusion_probs, var_probs, avg_coefs = calculate_bma_weights_and_coef(X_train, y_train, var_names_original)
            selected_indices = np.where(inclusion_probs > 0.15)[0]
            if len(selected_indices) == 0:
                selected_indices = np.arange(len(var_names_original))
            X_train_sel = X_train[:, selected_indices]
            X_test_sel = X_test[:, selected_indices]
            model = LinearRegression()
            model.fit(X_train_sel, y_train)
            y_pred_log = model.predict(X_test_sel)[0]

            bma_info = {
                'date': pred_date,
                'all_vars': var_names_original,
                'inclusion_probs': inclusion_probs.tolist(),
                'selected_vars': [var_names_original[idx] for idx in selected_indices],
                'n_selected': len(selected_indices)
            }
            bma_details[h].append(bma_info)

            # 存储系数信息
            coef_info = {
                'date': pred_date,
                'avg_coefs': avg_coefs,
                'var_names': var_names_original
            }
            coef_details[h].append(coef_info)

            # 存储概率矩阵（用于热力图和Excel）
            if prob_matrices[h]['var_names'] is None:
                prob_matrices[h]['var_names'] = var_names_original
            prob_matrices[h]['dates'].append(pred_date)
            prob_matrices[h]['matrix'].append(inclusion_probs)

        except Exception as e:
            print(f"  BMA失败，使用全变量: {e}")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)[0]
            bma_info = {
                'date': pred_date,
                'all_vars': var_names_original,
                'inclusion_probs': [1.0] * len(var_names_original),
                'selected_vars': var_names_original,
                'n_selected': len(var_names_original)
            }
            bma_details[h].append(bma_info)

            # 使用全变量模型的系数
            avg_coefs = model.coef_
            coef_info = {
                'date': pred_date,
                'avg_coefs': avg_coefs,
                'var_names': var_names_original
            }
            coef_details[h].append(coef_info)

            # 存储概率矩阵（全选中）
            if prob_matrices[h]['var_names'] is None:
                prob_matrices[h]['var_names'] = var_names_original
            prob_matrices[h]['dates'].append(pred_date)
            prob_matrices[h]['matrix'].append(np.ones(len(var_names_original)))

        y_pred_rv = np.expm1(y_pred_log)
        predictions[h]['dates'].append(pred_date)
        predictions[h]['pred'].append(y_pred_rv)
        predictions[h]['true'].append(true_rv)

print(f"BMA预测完成，耗时：{time.time() - start_time:.2f}秒")

# ----------------------
# 4. 导出BMA概率矩阵到Excel（核心功能）
# ----------------------
print("\n=== 导出BMA概率矩阵到Excel ===")

# 使用ExcelWriter并启用openpyxl引擎
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 统计指标
    stats_results = []
    for h in forecast_horizons:
        y_true = np.array(predictions[h]['true'])
        y_pred = np.array(predictions[h]['pred'])
        mse, mae, qlike, r2log = calculate_loss(y_true, y_pred)
        avg_selected = np.mean([info['n_selected'] for info in bma_details[h]])

        stats_results.append({
            "模型名称": model_name,
            "预测步数": horizon_names[forecast_horizons.index(h)],
            "MSE": round(mse, 8),
            "MAE": round(mae, 6),
            "QLIKE": round(qlike, 6),
            "R²LOG": round(r2log, 6),
            "平均选中变量数": round(avg_selected, 2)
        })
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_excel(writer, sheet_name="统计指标汇总", index=False)

    # 变量包含概率汇总（频率统计）
    inclusion_prob_results = []
    for h in forecast_horizons:
        horizon_name = horizon_names[forecast_horizons.index(h)]
        all_vars_dict = {}
        for info in bma_details[h]:
            for var, prob in zip(info['all_vars'], info['inclusion_probs']):
                all_vars_dict.setdefault(var, []).append(prob)
        for var, probs in all_vars_dict.items():
            avg_prob = np.mean(probs) * 100
            inclusion_prob_results.append({
                "模型名称": model_name,
                "预测步数": horizon_name,
                "变量名称": var,
                "平均包含概率(%)": round(avg_prob, 2)
            })
    inclusion_prob_df = pd.DataFrame(inclusion_prob_results)
    inclusion_prob_df.to_excel(writer, sheet_name="变量包含概率汇总", index=False)

    # 系数正负统计
    coef_sign_summary = []
    for h in forecast_horizons:
        horizon_name = horizon_names[forecast_horizons.index(h)]

        var_positive_count = {}
        var_negative_count = {}
        var_total_count = {}

        for record in coef_details[h]:
            avg_coefs = record['avg_coefs']
            var_names = record['var_names']

            for idx, var_name in enumerate(var_names):
                coef_value = avg_coefs[idx]

                if var_name not in var_total_count:
                    var_positive_count[var_name] = 0
                    var_negative_count[var_name] = 0
                    var_total_count[var_name] = 0

                if coef_value > 0:
                    var_positive_count[var_name] += 1
                elif coef_value < 0:
                    var_negative_count[var_name] += 1

                var_total_count[var_name] += 1

        for var_name in var_total_count.keys():
            total = var_total_count[var_name]
            pos_count = var_positive_count.get(var_name, 0)
            neg_count = var_negative_count.get(var_name, 0)
            pos_ratio = (pos_count / total * 100) if total > 0 else 0
            neg_ratio = (neg_count / total * 100) if total > 0 else 0

            if pos_ratio >= 60:
                main_sign = "正"
            elif neg_ratio >= 60:
                main_sign = "负"
            else:
                main_sign = "混合"

            coef_sign_summary.append({
                "模型名称": model_name,
                "预测步数": horizon_name,
                "变量名称": var_name,
                "选中总次数": total,
                "正系数次数": pos_count,
                "负系数次数": neg_count,
                "正系数占比(%)": round(pos_ratio, 2),
                "负系数占比(%)": round(neg_ratio, 2),
                "主要符号": main_sign
            })

    coef_sign_df = pd.DataFrame(coef_sign_summary)
    coef_sign_df = coef_sign_df.sort_values(["预测步数", "变量名称"])
    coef_sign_df.to_excel(writer, sheet_name="系数正负统计", index=False)

    # 符号汇总统计
    sign_summary_by_step = []
    for h in forecast_horizons:
        horizon_name = horizon_names[forecast_horizons.index(h)]
        subset = coef_sign_df[coef_sign_df["预测步数"] == horizon_name]

        if len(subset) > 0:
            n_positive_dominant = len(subset[subset["主要符号"] == "正"])
            n_negative_dominant = len(subset[subset["主要符号"] == "负"])
            n_mixed = len(subset[subset["主要符号"] == "混合"])
            avg_pos_ratio = subset["正系数占比(%)"].mean()
            avg_neg_ratio = subset["负系数占比(%)"].mean()

            top_positive = subset.nlargest(5, "正系数占比(%)")[["变量名称", "正系数占比(%)"]]
            top_positive_list = [f"{row['变量名称']}({row['正系数占比(%)']}%)" for _, row in top_positive.iterrows()]

            top_negative = subset.nlargest(5, "负系数占比(%)")[["变量名称", "负系数占比(%)"]]
            top_negative_list = [f"{row['变量名称']}({row['负系数占比(%)']}%)" for _, row in top_negative.iterrows()]

            sign_summary_by_step.append({
                "预测步数": horizon_name,
                "总变量数": len(subset),
                "主要为正的变量数": n_positive_dominant,
                "主要为负的变量数": n_negative_dominant,
                "符号混合的变量数": n_mixed,
                "平均正系数占比(%)": round(avg_pos_ratio, 2),
                "平均负系数占比(%)": round(avg_neg_ratio, 2),
                "正系数占比TOP5": "; ".join(top_positive_list) if top_positive_list else "无",
                "负系数占比TOP5": "; ".join(top_negative_list) if top_negative_list else "无"
            })

    sign_summary_df = pd.DataFrame(sign_summary_by_step)
    sign_summary_df.to_excel(writer, sheet_name="符号汇总统计", index=False)

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

    # 导出BMA概率矩阵（带颜色渐变格式的表格）
    print("\n正在生成带格式的BMA概率表格...")
    export_bma_probability_to_excel(writer, prob_matrices, full_var_names, horizon_names, forecast_horizons)

print(f"\n✅ 所有结果已保存至：{output_path}")

# ----------------------
# 5. 绘制BMA概率热力图（可选）
# ----------------------
print("\n=== 绘制BMA变量选择热力图 ===")

for h_idx, h in enumerate(forecast_horizons):
    horizon_name = horizon_names[h_idx]

    if len(prob_matrices[h]['matrix']) > 0:
        # 转换为numpy数组
        prob_matrix_original = np.array(prob_matrices[h]['matrix'])
        var_names_original = prob_matrices[h]['var_names']
        dates = prob_matrices[h]['dates']

        # 扩展到完整的变量列表
        prob_matrix_full = create_prob_matrix_for_all_lags(
            prob_matrix_original, var_names_original, full_var_names
        )

        # 使用完整变量名
        full_var_names_display = full_var_names

        # 绘制彩色概率热力图
        title = f'({horizon_name}) {model_name} - BMA变量包含概率随时间变化'
        filename = f'{fig_save_path}/{model_name}_{horizon_name}_BMA_selection.png'
        plot_bma_probability_heatmap(prob_matrix_full, full_var_names_display, dates, title, filename)

        print(f"已生成：{model_name} - {horizon_name} BMA彩色热力图")

# ----------------------
# 6. 输出汇总
# ----------------------
print(f"\n📊 可视化结果已保存至：{fig_save_path}")
print(f"   包括 {len(forecast_horizons)} 张BMA彩色变量选择热力图")
print("\n=== BMA变量包含概率结果汇总 ===")
print(inclusion_prob_df.head(12))
print(f"\n📊 统计指标汇总：")
print(stats_df)

print("\n=== 系数正负统计（前10行）===")
print(coef_sign_df[["预测步数", "变量名称", "正系数占比(%)", "负系数占比(%)", "主要符号"]].head(10))

print("\n=== 符号汇总统计 ===")
print(sign_summary_df)

# 验证滞后项逻辑
print("\n📋 变量维度验证：")
for h in forecast_horizons:
    n_lags = len(horizon_lag_map[h])
    rv_dim = n_lags
    exog_dim = len(all_exog_names) * n_lags
    total_dim = rv_dim + exog_dim
    print(f"{model_name}（{h}步）：RV维度{rv_dim} + 外生维度{exog_dim} = 总维度{total_dim}")

print("\n✅ BMA-HAR分析完成！")
print("\n💡 提示：")
print("   1. Excel文件中包含详细的BMA概率矩阵表格（带颜色渐变）")
print("   2. 绿色越深表示包含概率越高")
print("   3. 可以在Excel中自由调整字体大小、颜色等格式")
print("   4. 热力图图片保存在可视化文件夹中作为参考")