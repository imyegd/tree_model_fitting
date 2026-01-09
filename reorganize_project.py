#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目重构脚本：将项目分成束流数据和束位监测数据两部分
"""

import os
import shutil
from pathlib import Path

# 定义文件映射关系
BEAM_CURRENT_FILES = [
    'linear.py',
    'tree_model.py',
    'mlp.py',
    'LSTM.py',
    'PLS.py',
    'data_process.py',
]

BEAM_POSITION_FILES = [
    '01_data_prep.py',
    '02_baseline_models.py',
    'position.py',
    'plot.py',
    'XGboost.py',
    'PLS_test.py',
]

UTILS_FILES = [
    'convert_excel_to_csv.py',
    'rename_columns.py',
    'test.py',
    'MSPC.py',
    '03_pytorch_lstm.py',
]

def reorganize_files():
    """重新组织文件结构"""
    src_dir = Path('src')
    
    # 创建目标目录
    beam_current_dir = src_dir / 'beam_current'
    beam_position_dir = src_dir / 'beam_position'
    utils_dir = src_dir / 'utils'
    common_dir = src_dir / 'common'
    
    for dir_path in [beam_current_dir, beam_position_dir, utils_dir, common_dir]:
        dir_path.mkdir(exist_ok=True)
        # 创建 __init__.py
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""模块初始化"""\n', encoding='utf-8')
    
    # 移动束流数据相关文件
    print("移动束流数据相关文件...")
    for file_name in BEAM_CURRENT_FILES:
        src_file = src_dir / file_name
        if src_file.exists():
            dst_file = beam_current_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {file_name} -> beam_current/")
    
    # 移动束位监测数据相关文件
    print("\n移动束位监测数据相关文件...")
    for file_name in BEAM_POSITION_FILES:
        src_file = src_dir / file_name
        if src_file.exists():
            dst_file = beam_position_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {file_name} -> beam_position/")
    
    # 移动工具文件
    print("\n移动工具文件...")
    for file_name in UTILS_FILES:
        src_file = src_dir / file_name
        if src_file.exists():
            dst_file = utils_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {file_name} -> utils/")
    
    # utils.py 已经移动到 common/utils.py，这里不需要再移动
    print("\nutils.py 已移动到 common/utils.py")
    
    print("\n文件重组完成！")
    print("\n注意：")
    print("1. 原文件已保留在 src/ 目录")
    print("2. 请检查新目录中的文件，并更新导入路径")
    print("3. 更新完成后，可以删除 src/ 目录中的原文件")

if __name__ == '__main__':
    reorganize_files()
