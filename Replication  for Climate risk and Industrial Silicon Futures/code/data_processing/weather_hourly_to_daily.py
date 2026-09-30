import pandas as pd
import numpy as np
from datetime import datetime
import os

def hourly_to_daily_meteorological_data(file_path):
    """
    将小时级气象数据转换为日度平均数据
    :param file_path: 原始CSV文件路径
    :return: 日度平均数据DataFrame
    """
    # 1. 读取原始数据（处理文件格式，跳过前4行标题行）
    print("正在读取数据...")
    # 定义列名（根据数据结构确定）
    column_names = ['time', 'temperature_2m (°C)', 'relativehumidity_2m (%)', 
                   'cloudcover (%)', 'windspeed_10m (m/s)', 'unused']
    
    # 跳过前4行（包含冗余标题），从第5行开始读取实际数据
    df_hourly = pd.read_csv(
        file_path,
        skiprows=4,  # 关键：跳过前4行标题信息
        header=None, 
        names=column_names,
        encoding='utf-8'  # 避免中文编码问题
    )
    
    # 删除无用列
    df_hourly = df_hourly.drop('unused', axis=1)
    
    # 2. 数据预处理（时间格式转换+数值类型转换）
    print("正在处理数据格式...")
    # 转换时间格式（您的数据时间格式为：YYYY-MM-DDTHH:MM，如2022-12-22T00:00）
    df_hourly['time'] = pd.to_datetime(
        df_hourly['time'],
        format='%Y-%m-%dT%H:%M'  # 明确指定时间格式，确保转换正确
    )
    
    # 提取日期（用于按日分组）
    df_hourly['date'] = df_hourly['time'].dt.date
    
    # 转换数值列类型（确保为数值型用于计算）
    numeric_columns = ['temperature_2m (°C)', 'relativehumidity_2m (%)', 
                      'cloudcover (%)', 'windspeed_10m (m/s)']
    
    for col in numeric_columns:
        df_hourly[col] = pd.to_numeric(
            df_hourly[col],
            errors='coerce'  # 自动处理异常值，转为NaN后在计算时忽略
        )
    
    # 3. 按日期分组计算日平均值
    print("正在计算日度平均值...")
    df_daily = df_hourly.groupby('date').agg({
        'temperature_2m (°C)': 'mean',    # 日均温度
        'relativehumidity_2m (%)': 'mean',# 日均相对湿度
        'cloudcover (%)': 'mean',         # 日均云量
        'windspeed_10m (m/s)': 'mean'     # 日均风速
    }).round(2)  # 保留2位小数，保证数据精度
    
    # 重置索引，将date从索引转为普通列
    df_daily = df_daily.reset_index()
    
    # 格式化日期列为字符串（YYYY-MM-DD），便于后续使用
    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.strftime('%Y-%m-%d')
    
    # 4. 数据验证与信息输出
    print("="*50)
    print("数据处理完成！")
    print(f"原始数据量：{len(df_hourly)} 条小时级记录")
    print(f"日度数据量：{len(df_daily)} 条日度记录")
    print(f"时间范围：{df_daily['date'].iloc[0]} 至 {df_daily['date'].iloc[-1]}")
    print("="*50)
    
    return df_daily

def save_daily_data(df_daily, output_dir):
    """
    保存日度数据到Excel和CSV文件
    :param df_daily: 日度平均数据DataFrame
    :param output_dir: 输出文件夹路径
    """
    # 创建输出文件夹（若不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 定义输出文件名（包含时间戳，避免重复）
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(output_dir, f'北京气象日度数据_{current_time}.xlsx')
    csv_path = os.path.join(output_dir, f'北京气象日度数据_{current_time}.csv')
    
    # 保存为Excel文件（包含数据说明）
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 保存日度数据
        df_daily.to_excel(writer, sheet_name='日度平均数据', index=False)
        
        # 保存数据说明
        explanation_data = {
            '说明项目': ['数据来源', '数据时间范围', '数据类型', '数据量', 
                       '温度单位', '湿度单位', '云量单位', '风速单位', '计算方法'],
            '详细说明': [
                '北京三年小时级气象观测数据',
                f'{df_daily["date"].iloc[0]} 至 {df_daily["date"].iloc[-1]}',
                '日度平均值（温度、湿度、云量、风速）',
                f'{len(df_daily)} 条日度记录',
                '摄氏度（°C）', '百分比（%）', '百分比（%）', '米/秒（m/s）',
                '每日24小时数据算术平均值，保留2位小数'
            ]
        }
        df_explanation = pd.DataFrame(explanation_data)
        df_explanation.to_excel(writer, sheet_name='数据说明', index=False)
    
    # 保存为CSV文件（兼容更多软件）
    df_daily.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig支持中文显示
    
    print(f"\n文件已保存至：{output_dir}")
    print(f"Excel文件：{os.path.basename(excel_path)}")
    print(f"CSV文件：{os.path.basename(csv_path)}")

# ---------------------- 主程序运行入口 ----------------------
if __name__ == "__main__":
    # 1. 设置文件路径（请确认此路径与您电脑中的实际路径一致！）
    # 您提供的路径：C:\Users\35632\Desktop\论文\气象数据\北京\history_20260204T114839.csv
    input_file_path = r"C:\Users\35632\Desktop\论文\气象数据\北京\history_20260204T114839.csv"
    
    # 2. 设置输出文件夹路径（建议与输入文件同目录，便于查找）
    output_directory = r"C:\Users\35632\Desktop\论文\气象数据\北京\日度数据结果"
    
    # 3. 执行数据转换
    try:
        # 转换小时数据为日度数据
        daily_data = hourly_to_daily_meteorological_data(input_file_path)
        
        # 显示前5条数据预览
        print("\n日度数据预览（前5条）：")
        print(daily_data.head().to_string(index=False))
        
        # 保存结果文件
        save_daily_data(daily_data, output_directory)
        
        print("\n✅ 所有操作完成！您可在输出文件夹中查看结果文件。")
    
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        print("请检查以下事项：")
        print("1. 输入文件路径是否正确（文件是否存在）")
        print("2. Excel文件是否被其他程序占用")
        print("3. 是否已安装所需Python库（pandas、openpyxl）")