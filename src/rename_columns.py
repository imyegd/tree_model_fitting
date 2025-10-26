#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改CSV文件列名的脚本
"""

import pandas as pd
import os

def rename_csv_columns(csv_file_path):
    """
    修改CSV文件的列名
    将"特征1", "特征2", ... 改为 "feature1", "feature2", ...
    将"束流"改为"target"
    """
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        print(f"原始列名: {list(df.columns)}")
        
        # 创建列名映射字典
        column_mapping = {}
        
        for col in df.columns:
            if col == "束流":
                column_mapping[col] = "target"
            elif col.startswith("特征"):
                # 提取特征编号
                feature_num = col.replace("特征", "")
                column_mapping[col] = f"feature{feature_num}"
            # 其他列保持不变（如"时间"列）
        
        print(f"列名映射: {column_mapping}")
        
        # 重命名列
        df_renamed = df.rename(columns=column_mapping)
        
        print(f"修改后的列名: {list(df_renamed.columns)}")
        
        # 保存修改后的文件
        df_renamed.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"已保存修改后的文件: {csv_file_path}")
        
        return True
        
    except Exception as e:
        print(f"修改列名时出现错误: {str(e)}")
        return False

def main():
    # CSV文件路径
    csv_file = "data/束位监测数据.csv"
    
    # 执行列名修改
    success = rename_csv_columns(csv_file)
    
    if success:
        print("列名修改完成！")
    else:
        print("列名修改失败")

if __name__ == "__main__":
    main()
