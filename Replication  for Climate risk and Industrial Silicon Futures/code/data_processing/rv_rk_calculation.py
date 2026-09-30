import pandas as pd
import numpy as np
import os


def calculate_rv_rk(file_paths, output_path):
    """
    计算每日RV（已实现波动率）和RK（已实现核，Realized Kernel）
    严格按照论文公式实现：
    RV = sum(r^2)  （方差形式）
    RK = sum(r^2) + 2 * sum_{w=1}^q (1 - w/(q+1)) * sum_{i=1}^{n-w} r_i * r_{i+w}  （方差形式）
    :param file_paths: 输入CSV文件路径列表
    :param output_path: 输出结果路径
    """
    # 读取并合并所有CSV文件（日期不重复）
    df_list = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"警告：文件 {path} 不存在，已跳过")
            continue

        # 尝试多种编码读取
        try:
            df = pd.read_csv(path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding='gb2312')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='utf-8-sig')

        # 数据预处理
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.dropna(subset=['日期', '收盘价'])
        df_list.append(df)
        print(f"已读取文件：{path}，数据量：{len(df)}行")

    if not df_list:
        print("错误：未读取到有效数据文件")
        return

    # 合并所有数据
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values('日期')

    # 计算5分钟对数收益率
    df_all['对数收盘价'] = np.log(df_all['收盘价'])
    df_all['收益率'] = df_all['对数收盘价'].diff()
    df_all = df_all.dropna(subset=['收益率'])

    # 按日期分组计算每日RV和RK
    daily_results = []
    grouped = df_all.groupby(df_all['日期'].dt.date)

    for date, group in grouped:
        returns = group['收益率'].values
        n = len(returns)

        if n < 2:
            continue

        # 1. 计算RV（已实现波动率）：收益率平方和（方差形式）
        sum_sq = np.sum(returns ** 2)
        rv_var = sum_sq  # 论文定义的RV（方差形式）

        # 2. 计算RK（已实现核）
        # 设置参数q
        omega = 30  # 期望滞后窗口宽度（分钟）
        sampling_length = 240  # 采样周期长度（分钟）
        q = int(np.ceil((omega / sampling_length) * n))

        if q >= n:
            q = n - 1

        # 计算滞后项加权和
        lag_sum = 0.0
        for w in range(1, q + 1):
            weight = 1 - (w / (q + 1))
            cross_sum = np.sum(returns[:-w] * returns[w:])
            lag_sum += weight * cross_sum

        # RK（方差形式）
        rk_var = sum_sq + 2 * lag_sum

        # 3. 计算标准差形式（供参考，不用于主分析）
        rv_std = np.sqrt(rv_var) if rv_var > 0 else 0
        rk_std = np.sqrt(rk_var) if rk_var > 0 else 0

        daily_results.append({
            '日期': pd.to_datetime(date),
            '交易时段数': n,
            'RV(方差形式)': rv_var,  # 主分析用（论文定义）
            'RK(方差形式)': rk_var,  # 主分析用（论文定义）
            'RV(标准差形式)': rv_std,  # 参考用
            'RK(标准差形式)': rk_std,  # 参考用
            'q参数': q
        })

    # 整理结果并保存
    result_df = pd.DataFrame(daily_results)
    result_df = result_df.sort_values('日期').reset_index(drop=True)

    # 保存到CSV文件
    result_df.to_csv(output_path, index=False, encoding='gbk')
    print(f"\n计算完成！结果已保存到：{output_path}")
    print(f"共覆盖 {len(result_df)} 个交易日")
    print(f"日期范围：{result_df['日期'].min().strftime('%Y-%m-%d')} 至 {result_df['日期'].max().strftime('%Y-%m-%d')}")

    # 输出统计摘要
    print(f"\n统计摘要（方差形式，用于主分析）：")
    print(f"RV(方差形式)均值：{result_df['RV(方差形式)'].mean():.8f}")
    print(f"RV(方差形式)标准差：{result_df['RV(方差形式)'].std():.8f}")
    print(f"RK(方差形式)均值：{result_df['RK(方差形式)'].mean():.8f}")
    print(f"RK(方差形式)标准差：{result_df['RK(方差形式)'].std():.8f}")

    print(f"\n统计摘要（标准差形式，仅供参考）：")
    print(f"RV(标准差形式)均值：{result_df['RV(标准差形式)'].mean():.6f}")
    print(f"RV(标准差形式)标准差：{result_df['RV(标准差形式)'].std():.6f}")
    print(f"RK(标准差形式)均值：{result_df['RK(标准差形式)'].mean():.6f}")
    print(f"RK(标准差形式)标准差：{result_df['RK(标准差形式)'].std():.6f}")


if __name__ == "__main__":
    desktop_path = r"C:\Users\35632\Desktop"
    file_names = [
        "gzsi2308.csv",
        "gzsi0001.csv",
        "24gzsi0001.csv",
        "25gzsi0001.csv"
    ]

    file_paths = [os.path.join(desktop_path, name) for name in file_names]
    output_file = os.path.join(desktop_path, "三年RV_RK计算结果_方差形式.csv")

    calculate_rv_rk(file_paths, output_file)