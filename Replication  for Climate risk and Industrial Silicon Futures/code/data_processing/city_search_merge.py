import pandas as pd
import os
from pathlib import Path

def merge_city_search_data(root_folder, keyword, output_file):
    """
    整合单个关键词（单晶硅/多晶硅）的所有城市搜索量数据
    :param root_folder: 包含所有城市Excel文件的根文件夹路径
    :param keyword: 关键词（"单晶硅"或"多晶硅"）
    :param output_file: 输出Excel文件路径
    """
    # 存储所有城市的搜索量数据（key：城市名，value：数据Series）
    city_data_dict = {}
    # 存储共用的日期列（以第一个城市为准）
    common_dates = None
    
    print(f"正在读取{keyword}相关文件...")
    # 遍历根文件夹下所有Excel文件
    for file_name in os.listdir(root_folder):
        # 筛选当前关键词的文件（文件名包含关键词）
        if file_name.endswith('.xlsx') and keyword in file_name:
            # 提取城市名（从文件名中截取，格式："关键词_城市.xlsx" → 提取"城市"）
            city_name = file_name.replace(f'{keyword}_', '').replace('.xlsx', '')
            if not city_name or city_name == file_name:
                # 若文件名格式异常，用文件名（不含后缀）作为城市名
                city_name = Path(file_name).stem
            print(f"正在处理：{city_name} - {file_name}")
            
            # 读取Excel文件
            file_path = os.path.join(root_folder, file_name)
            df_city = pd.read_excel(file_path, sheet_name=0)
            
            # 数据验证：确保包含必要字段
            required_cols = ['时间', '搜索pc+移动']
            if not all(col in df_city.columns for col in required_cols):
                # 兼容可能的字段名差异（如"搜索pc+移动"可能为"搜索pc+移动 "带空格）
                df_city.columns = df_city.columns.str.strip()  # 去除列名空格
                if not all(col in df_city.columns for col in required_cols):
                    print(f"⚠️  {city_name} 文件缺少必要字段，跳过该文件")
                    continue
            
            # 标准化日期格式（统一为YYYY-MM-DD）
            df_city['时间'] = pd.to_datetime(df_city['时间']).dt.strftime('%Y-%m-%d')
            
            # 保存共用日期列（以第一个有效城市的日期为准）
            if common_dates is None:
                common_dates = df_city['时间'].copy()
            else:
                # 对齐日期，缺失数据填充为NaN
                df_city = df_city.merge(
                    pd.DataFrame({'时间': common_dates}),
                    on='时间',
                    how='right'
                )
            
            # 提取PC+移动搜索量列，并重命名为城市名
            city_search_data = df_city.set_index('时间')['搜索pc+移动'].rename(city_name)
            city_data_dict[city_name] = city_search_data
    
    # 检查是否成功读取数据
    if not city_data_dict:
        print(f"❌ 未找到{keyword}相关的有效城市数据文件")
        return
    
    print(f"\n成功读取 {len(city_data_dict)} 个城市的{keyword}数据")
    
    # 合并所有城市数据（按日期索引合并）
    print("正在合并所有城市数据...")
    merged_df = pd.DataFrame(index=common_dates)
    for city_name, city_series in city_data_dict.items():
        merged_df[city_name] = city_series
    
    # 重置索引，将日期转为列
    merged_df = merged_df.reset_index().rename(columns={'index': '日期'})
    
    # 保存到Excel文件
    print(f"正在保存到Excel文件：{output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name=f'{keyword}搜索量', index=False)
        
        # 添加数据说明工作表
        explanation_data = {
            '说明项目': [
                '数据来源', '城市数量', '数据时间范围', '包含字段', '数据格式说明', '缺失值处理'
            ],
            '详细说明': [
                f'各城市{keyword}搜索量Excel文件',
                f'{len(city_data_dict)} 个城市',
                f'{common_dates.iloc[0]} 至 {common_dates.iloc[-1]}',
                '日期（YYYY-MM-DD）、各城市PC+移动端搜索量',
                '每行对应一个日期，每列对应一个城市的搜索量',
                '不同城市日期不一致时，缺失数据填充为NaN'
            ]
        }
        pd.DataFrame(explanation_data).to_excel(writer, sheet_name='数据说明', index=False)
    
    print(f"✅ {keyword}数据整合完成！文件已保存至：{output_file}")
    print(f"📊 整合结果概览：")
    print(f"   - 城市数量：{len(city_data_dict)} 个")
    print(f"   - 时间跨度：{len(common_dates)} 天")
    print(f"   - 数据维度：{len(merged_df.columns)} 列（1列日期 + {len(merged_df.columns)-1} 列城市数据）")

# ---------------------- 主程序运行入口 ----------------------
if __name__ == "__main__":
    # 1. 设置文件夹路径（请根据实际情况确认！）
    # 根文件夹：包含所有城市的单晶硅、多晶硅Excel文件
    root_folder = r"C:\Users\35632\Desktop\论文\指数权重"
    
    # 2. 定义两个关键词及输出文件名
    keywords_output = [
        ("单晶硅", "所有城市单晶硅搜索量整合.xlsx"),
        ("多晶硅", "所有城市多晶硅搜索量整合.xlsx")
    ]
    
    # 3. 执行整合（分别处理两个关键词）
    for keyword, output_filename in keywords_output:
        output_file = os.path.join(root_folder, output_filename)
        merge_city_search_data(root_folder, keyword, output_file)
        print("="*60)
    
    print(f"\n🎉 所有关键词数据整合完毕！")
    print(f"📁 输出文件位置：{root_folder}")
    print(f"📄 生成文件：")
    for _, output_filename in keywords_output:
        print(f"   - {output_filename}")