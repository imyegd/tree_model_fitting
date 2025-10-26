#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Excel文件转换为CSV文件的脚本
"""

import pandas as pd
import os
import sys

def convert_excel_to_csv(excel_file_path, output_dir=None):
    """
    将Excel文件转换为CSV文件
    
    Args:
        excel_file_path (str): Excel文件路径
        output_dir (str): 输出目录，如果为None则使用Excel文件所在目录
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(excel_file_path):
            print(f"错误：文件 {excel_file_path} 不存在")
            return False
        
        # 读取Excel文件
        print(f"正在读取Excel文件: {excel_file_path}")
        
        # 读取所有工作表
        excel_file = pd.ExcelFile(excel_file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"发现 {len(sheet_names)} 个工作表: {sheet_names}")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(excel_file_path)
        
        # 为每个工作表创建CSV文件
        for sheet_name in sheet_names:
            print(f"正在处理工作表: {sheet_name}")
            
            # 读取工作表数据
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(excel_file_path))[0]
            if len(sheet_names) > 1:
                csv_filename = f"{base_name}_{sheet_name}.csv"
            else:
                csv_filename = f"{base_name}.csv"
            
            csv_file_path = os.path.join(output_dir, csv_filename)
            
            # 保存为CSV文件
            df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
            print(f"已保存: {csv_file_path}")
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
        
        print("转换完成！")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

def main():
    # Excel文件路径
    excel_file = "data\束位监测数据.xlsx"
    
    # 执行转换
    success = convert_excel_to_csv(excel_file)
    
    if success:
        print("Excel文件已成功转换为CSV格式")
    else:
        print("转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
