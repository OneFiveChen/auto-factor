import pandas as pd
import os
from datetime import datetime

# 2020年8月11日的时间戳（UTC时间）
FILTER_DATE = '2020-08-11'
FILTER_TIMESTAMP = datetime.strptime(FILTER_DATE, '%Y-%m-%d').timestamp()

def filter_csv_by_timestamp(file_path):
    """
    读取CSV文件，过滤掉指定时间戳之前的数据，并保存为新文件
    
    Args:
        file_path: CSV文件路径
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 检查是否存在time列
        if 'time' not in df.columns:
            print(f"错误: 文件 {file_path} 中不存在'time'列")
            return False
        
        # 过滤数据：保留time列大于等于FILTER_TIMESTAMP的数据
        print(f"过滤掉 {FILTER_DATE} 之前的数据...")
        filtered_df = df[df['time'] >= FILTER_TIMESTAMP]
        
        # 计算过滤掉的行数
        filtered_count = len(df) - len(filtered_df)
        print(f"共过滤掉 {filtered_count} 行数据")
        print(f"过滤后剩余 {len(filtered_df)} 行数据")
        
        # 生成输出文件名
        base_name, ext = os.path.splitext(file_path)
        output_file = f"{base_name}_filtered{ext}"
        
        # 保存过滤后的数据
        filtered_df.to_csv(output_file, index=False)
        print(f"过滤后的数据已保存至: {output_file}")
        
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def main():
    """
    主函数：处理当前目录下的所有CSV文件
    """
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 查找当前目录下的所有CSV文件
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        file_path = os.path.join(current_dir, csv_file)
        print(f"\n处理文件 {csv_file}...")
        filter_csv_by_timestamp(file_path)
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main()