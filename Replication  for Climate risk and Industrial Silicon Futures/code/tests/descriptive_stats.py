# -*- coding: utf-8 -*-
"""
输出类似论文Table 3的描述性统计表
包含：均值、标准差、偏度、峰度、JB检验、LB检验(Q5,Q10,Q20)、ADF单位根检验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import openpyxl
import os
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ----------------------
# 1. 文件路径配置
# ----------------------
data_path = r"C:\Users\35632\Desktop\总回归数据.xlsx"
output_path = r"C:\Users\35632\Desktop\Table3_完整统计_PCA结果.xlsx"
fig_save_path = r"C:\Users\35632\Desktop\论文\PCA_可视化"

# 创建输出目录
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)


# ----------------------
# 2. 工具函数
# ----------------------
def clean_percentage_column(series):
    """清理带百分号的列，转换为数值"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace('%', '').str.strip()
        series = pd.to_numeric(series, errors='coerce')
    return series


def calculate_stats_with_tests(series, name):
    """
    计算变量的描述性统计量和各项检验
    """
    series = clean_percentage_column(series)
    series = pd.to_numeric(series, errors='coerce')
    series = series.dropna()

    if len(series) < 10:
        return {
            '变量': name,
            '样本数': len(series),
            '均值': np.nan,
            '标准差': np.nan,
            '最小值': np.nan,
            '最大值': np.nan,
            '偏度': np.nan,
            '峰度': np.nan,
            'JB统计量': np.nan,
            'JB显著性': '',
            'LB Q(5)统计量': np.nan,
            'LB Q(5)显著性': '',
            'LB Q(10)统计量': np.nan,
            'LB Q(10)显著性': '',
            'LB Q(20)统计量': np.nan,
            'LB Q(20)显著性': '',
            'ADF统计量': np.nan,
            'ADF显著性': ''
        }

    mean_val = series.mean()
    std_val = series.std()
    min_val = series.min()
    max_val = series.max()
    skewness = stats.skew(series)
    kurtosis = stats.kurtosis(series)

    # Jarque-Bera检验
    jb_stat, jb_p = stats.jarque_bera(series)
    jb_sig = ''
    if jb_p < 0.01:
        jb_sig = '***'
    elif jb_p < 0.05:
        jb_sig = '**'
    elif jb_p < 0.1:
        jb_sig = '*'

    # Ljung-Box检验
    series_clean = series.dropna().values
    lb_results = {}
    for lag in [5, 10, 20]:
        if len(series_clean) > lag:
            try:
                lb_result = acorr_ljungbox(series_clean, lags=[lag], return_df=True)
                lb_stat = lb_result.iloc[0, 0]
                lb_p = lb_result.iloc[0, 1]
            except:
                lb_stat, lb_p = acorr_ljungbox(series_clean, lags=[lag], return_df=False)[0]

            lb_sig = ''
            if lb_p < 0.01:
                lb_sig = '***'
            elif lb_p < 0.05:
                lb_sig = '**'
            elif lb_p < 0.1:
                lb_sig = '*'
            lb_results[f'LB{lag}'] = (lb_stat, lb_sig)
        else:
            lb_results[f'LB{lag}'] = (np.nan, '')

    # ADF检验
    try:
        adf_result = adfuller(series_clean, autolag='AIC')
        adf_stat = adf_result[0]
        adf_p = adf_result[1]
        adf_sig = ''
        if adf_p < 0.01:
            adf_sig = '***'
        elif adf_p < 0.05:
            adf_sig = '**'
        elif adf_p < 0.1:
            adf_sig = '*'
    except:
        adf_stat = np.nan
        adf_sig = ''

    return {
        '变量': name,
        '样本数': len(series),
        '均值': mean_val,
        '标准差': std_val,
        '最小值': min_val,
        '最大值': max_val,
        '偏度': skewness,
        '峰度': kurtosis,
        'JB统计量': jb_stat,
        'JB显著性': jb_sig,
        'LB Q(5)统计量': lb_results['LB5'][0],
        'LB Q(5)显著性': lb_results['LB5'][1],
        'LB Q(10)统计量': lb_results['LB10'][0],
        'LB Q(10)显著性': lb_results['LB10'][1],
        'LB Q(20)统计量': lb_results['LB20'][0],
        'LB Q(20)显著性': lb_results['LB20'][1],
        'ADF统计量': adf_stat,
        'ADF显著性': adf_sig
    }


def pca_analysis(data, cols, n_components, name):
    """
    主成分分析 - 保持最初版本，不做额外处理
    """
    if data.shape[1] == 0:
        print(f"警告: {name} 没有有效数据，跳过PCA")
        return None, None, None

    # 标准化 - 如果数据有NaN，这里会报错，但我们先不做处理
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=min(n_components, data.shape[1]), random_state=42)
    data_pca = pca.fit_transform(data_scaled)

    # 载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'{name}_PC{i + 1}' for i in range(pca.n_components_)],
        index=cols
    )

    # 各变量对主成分的贡献度
    var_contribution = np.abs(loadings).sum(axis=1)
    var_contribution = var_contribution / var_contribution.sum() * 100
    loadings_df['总贡献度(%)'] = var_contribution.round(2)

    # 方差贡献率
    exp_var = pca.explained_variance_ratio_ * 100
    exp_var_df = pd.DataFrame({
        '主成分': [f'{name}_PC{i + 1}' for i in range(pca.n_components_)],
        '特征值': pca.explained_variance_,
        '方差贡献率(%)': exp_var.round(2),
        '累计贡献率(%)': np.cumsum(exp_var).round(2)
    })

    # 可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(loadings_df.iloc[:, :-1], cmap='RdBu_r', annot=True, fmt='.2f',
                cbar_kws={'label': '载荷系数'})
    plt.title(f'{name} 主成分载荷矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{name}_PCA载荷矩阵.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    x = range(1, pca.n_components_ + 1)
    plt.bar(x, exp_var, alpha=0.7, label='单个贡献率')
    plt.plot(x, np.cumsum(exp_var), 'ro-', label='累计贡献率')
    plt.xlabel('主成分')
    plt.ylabel('贡献率(%)')
    plt.title(f'{name} 主成分方差贡献率', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}/{name}_贡献率.png', dpi=300, bbox_inches='tight')
    plt.close()

    return data_pca, loadings_df, exp_var_df


# ----------------------
# 3. 加载数据
# ----------------------
print("正在加载数据...")
df = pd.read_excel(data_path, sheet_name=0)

# 处理日期列
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
print(f"总样本量: {len(df)} 天")
print(f"总变量数: {len(df.columns)}")

# 清理污染比例列
for col in df.columns:
    if '比例' in col:
        df[col] = clean_percentage_column(df[col])
        print(f"已清理列: {col}")

# ----------------------
# 4. 变量分组 - 回到最初的白名单方式
# ----------------------
all_cols = df.columns.tolist()
print("\n正在识别变量分组...")

# RV变量
rv_vars = ['rv'] if 'rv' in df.columns else []

# 平均空气污染变量
avgplt_vars = ['日均pm2.5', '日均pm10', '日均aqi',
               '严重污染比例', '重度污染比例', '中度污染比例', '轻度污染比例']

# 加权空气污染变量
weightplt_vars = ['单晶硅加权平均pm2.5', '单晶硅加权平均pm10', '单晶硅加权平均aqi',
                  '多晶硅加权平均pm2.5', '多晶硅加权平均pm10', '多晶硅加权平均aqi']

# 气象变量
weather_vars = ['单晶硅加权温度', '单晶硅加权风速', '单晶硅加权相对湿度',
                '单晶硅加权总云量', '多晶硅加权相对湿度', '多晶硅加权总云量',
                '主产区降水量', '总辐射']

# 气候变化关注度原始变量 - 明确的中文名单
clmfactor_raw_vars = [
    '气候', '大气', '气候变化', '气温', '二氧化碳', '辐射', '排放', '全球变暖',
    '碳', '京都议定书', '碳排放', '碳足迹', '温室', '温室气体', '巴黎协定',
    '化石燃料', '可再生能源', '联合国气候变化框架公约', '冰川融化', '海平面上升',
    '日照', '太阳辐射', '太阳能', '光伏', 'esg', '单晶硅', '多晶硅'
]

# 极端天气关注度原始变量 - 明确的中文名单
weafactor_raw_vars = [
    '热浪', '沙尘暴', '台风', '暴雨', '干旱', '洪水', '极寒', '雷暴', '高温',
    '寒潮', '极端天气', '野火', '海平面', '气象灾害'
]

# 只保留实际存在于数据中的变量
avgplt_vars = [v for v in avgplt_vars if v in df.columns]
weightplt_vars = [v for v in weightplt_vars if v in df.columns]
weather_vars = [v for v in weather_vars if v in df.columns]
clmfactor_raw_vars = [v for v in clmfactor_raw_vars if v in df.columns]
weafactor_raw_vars = [v for v in weafactor_raw_vars if v in df.columns]

print(f"RV变量: {rv_vars}")
print(f"平均空气污染变量: {avgplt_vars} (共{len(avgplt_vars)}个)")
print(f"加权空气污染变量: {weightplt_vars[:3]}... (共{len(weightplt_vars)}个)")
print(f"气象变量: {weather_vars[:3]}... (共{len(weather_vars)}个)")
print(f"气候变化关注度原始变量: {clmfactor_raw_vars[:5]}... (共{len(clmfactor_raw_vars)}个)")
print(f"极端天气关注度原始变量: {weafactor_raw_vars[:5]}... (共{len(weafactor_raw_vars)}个)")

# ----------------------
# 5. 计算描述性统计
# ----------------------
print("\n正在计算描述性统计和各项检验...")

stats_results = []

# 收集所有要统计的变量
all_vars_to_stats = (rv_vars + avgplt_vars + weightplt_vars + weather_vars +
                     clmfactor_raw_vars + weafactor_raw_vars)

for var in all_vars_to_stats:
    if var in df.columns:
        print(f"正在处理: {var}")
        stats_results.append(calculate_stats_with_tests(df[var], var))

# 转换为DataFrame
if stats_results:
    stats_df = pd.DataFrame(stats_results)
    numeric_cols = ['均值', '标准差', '最小值', '最大值', '偏度', '峰度',
                    'JB统计量', 'LB Q(5)统计量', 'LB Q(10)统计量', 'LB Q(20)统计量', 'ADF统计量']
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(4)
    print(f"\n描述性统计计算完成，共{len(stats_df)}个变量")
else:
    stats_df = pd.DataFrame()
    print("警告: 没有找到任何变量")

# ----------------------
# 6. 主成分分析
# ----------------------
print("\n开始主成分分析...")

pca_results = {}
all_pc_scores = []

# 6.1 气候变化关注度主成分分析
if len(clmfactor_raw_vars) >= 3:
    print(f"\n对气候变化关注度变量({len(clmfactor_raw_vars)}个)进行PCA...")
    # 检查数据是否有NaN
    X_clm = df[clmfactor_raw_vars].values
    if np.any(pd.isna(X_clm)):
        print(f"  警告: 数据中包含NaN，PCA可能失败。请检查数据完整性。")

    clm_pca_scores, clm_loadings, clm_exp_var = pca_analysis(
        X_clm, clmfactor_raw_vars, 3, 'CLMFactor'
    )

    if clm_pca_scores is not None:
        for i in range(clm_pca_scores.shape[1]):
            pc_name = f'CLMFactor_PC{i + 1}'
            df[pc_name] = clm_pca_scores[:, i]
            all_pc_scores.append(pc_name)

        pca_results['CLMFactor'] = {
            '载荷矩阵': clm_loadings,
            '方差贡献率': clm_exp_var
        }

# 6.2 极端天气关注度主成分分析
if len(weafactor_raw_vars) >= 3:
    print(f"\n对极端天气关注度变量({len(weafactor_raw_vars)}个)进行PCA...")
    X_wea = df[weafactor_raw_vars].values
    if np.any(pd.isna(X_wea)):
        print(f"  警告: 数据中包含NaN，PCA可能失败。请检查数据完整性。")

    wea_pca_scores, wea_loadings, wea_exp_var = pca_analysis(
        X_wea, weafactor_raw_vars, 3, 'WEAFactor'
    )

    if wea_pca_scores is not None:
        for i in range(wea_pca_scores.shape[1]):
            pc_name = f'WEAFactor_PC{i + 1}'
            df[pc_name] = wea_pca_scores[:, i]
            all_pc_scores.append(pc_name)

        pca_results['WEAFactor'] = {
            '载荷矩阵': wea_loadings,
            '方差贡献率': wea_exp_var
        }

# 6.3 计算主成分的描述性统计
for var in all_pc_scores:
    if var in df.columns:
        print(f"正在处理: {var}")
        stats_results.append(calculate_stats_with_tests(df[var], var))

# 更新stats_df
if stats_results:
    stats_df = pd.DataFrame(stats_results)
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(4)

# ----------------------
# 7. 保存结果
# ----------------------
print(f"\n正在保存结果到: {output_path}")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    if not stats_df.empty:
        stats_df.to_excel(writer, sheet_name="Table3_完整统计", index=False)

    if 'CLMFactor' in pca_results:
        pca_results['CLMFactor']['载荷矩阵'].to_excel(writer, sheet_name="CLMFactor_PCA载荷")
        pca_results['CLMFactor']['方差贡献率'].to_excel(writer, sheet_name="CLMFactor_PCA贡献率", index=False)

    if 'WEAFactor' in pca_results:
        pca_results['WEAFactor']['载荷矩阵'].to_excel(writer, sheet_name="WEAFactor_PCA载荷")
        pca_results['WEAFactor']['方差贡献率'].to_excel(writer, sheet_name="WEAFactor_PCA贡献率", index=False)

    if all_pc_scores:
        pc_scores_df = df[all_pc_scores].copy()
        pc_scores_df.to_excel(writer, sheet_name="主成分得分", index=True)

print(f"\n✅ 所有结果已保存至: {output_path}")
print(f"📊 PCA可视化结果已保存至: {fig_save_path}")