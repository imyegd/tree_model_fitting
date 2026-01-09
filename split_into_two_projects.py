#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将项目拆分为两个独立项目：beam_current 和 beam_position
"""

import os
import shutil
from pathlib import Path

def create_project_structure(project_dir, project_name):
    """创建项目目录结构"""
    dirs = [
        'src',
        'data/raw',
        'data/processed',
        'models',
        'result'
    ]
    for dir_path in dirs:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
    print(f"创建 {project_name} 项目目录结构完成")

def copy_beam_current_project():
    """创建束流数据项目"""
    project_dir = Path('beam_current')
    project_name = '束流数据'
    
    print(f"\n{'='*60}")
    print(f"创建 {project_name} 项目")
    print(f"{'='*60}")
    
    create_project_structure(project_dir, project_name)
    
    # 1. 复制代码文件
    print("\n1. 复制代码文件...")
    src_files = [
        'src/beam_current/linear.py',
        'src/beam_current/tree_model.py',
        'src/beam_current/mlp.py',
        'src/beam_current/LSTM.py',
        'src/beam_current/PLS.py',
        'src/beam_current/data_process.py',
    ]
    
    for src_file in src_files:
        if Path(src_file).exists():
            dst_file = project_dir / 'src' / Path(src_file).name
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {Path(src_file).name}")
    
    # 2. 复制数据文件
    print("\n2. 复制数据文件...")
    data_files = [
        'data/raw/束流.csv',
        'data/raw/束流.xlsx',
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            dst_file = project_dir / 'data' / 'raw' / Path(data_file).name
            shutil.copy2(data_file, dst_file)
            print(f"  复制: {Path(data_file).name}")
    
    # 3. 复制处理后的数据（如果有）
    processed_files = [
        'data/processed/scaler.pkl',
        'data/processed/split_data.npz',
    ]
    
    for proc_file in processed_files:
        if Path(proc_file).exists():
            dst_file = project_dir / 'data' / 'processed' / Path(proc_file).name
            shutil.copy2(proc_file, dst_file)
            print(f"  复制: {Path(proc_file).name}")
    
    # 4. 复制结果文件
    print("\n3. 复制结果文件...")
    result_dir = Path('result/束流')
    if result_dir.exists():
        dst_result = project_dir / 'result'
        if dst_result.exists():
            shutil.rmtree(dst_result)
        shutil.copytree(result_dir, dst_result)
        print(f"  复制结果目录: {result_dir}")
    
    # 5. 复制模型文件（如果有）
    print("\n4. 复制模型文件...")
    model_files = [
        'models/best_lstm_model.pth',
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            dst_file = project_dir / 'models' / Path(model_file).name
            shutil.copy2(model_file, dst_file)
            print(f"  复制: {Path(model_file).name}")
    
    # 6. 创建 requirements.txt
    print("\n5. 创建 requirements.txt...")
    requirements = """jupyter>=1.1.1
lightgbm>=4.6.0
matplotlib>=3.10.7
numpy>=2.3.4
openpyxl>=3.1.5
pandas>=2.3.3
scikit-learn>=1.7.2
scipy>=1.16.2
seaborn>=0.13.2
tqdm>=4.67.1
xgboost>=3.1.1
torch>=2.0.0
"""
    (project_dir / 'requirements.txt').write_text(requirements, encoding='utf-8')
    print("  创建 requirements.txt")
    
    print(f"\n{project_name} 项目创建完成！")

def copy_beam_position_project():
    """创建束位监测数据项目"""
    project_dir = Path('beam_position')
    project_name = '束位监测数据'
    
    print(f"\n{'='*60}")
    print(f"创建 {project_name} 项目")
    print(f"{'='*60}")
    
    create_project_structure(project_dir, project_name)
    
    # 1. 复制代码文件
    print("\n1. 复制代码文件...")
    src_files = [
        'src/beam_position/01_data_prep.py',
        'src/beam_position/02_baseline_models.py',
        'src/beam_position/position.py',
        'src/beam_position/plot.py',
        'src/beam_position/XGboost.py',
        'src/beam_position/PLS_test.py',
    ]
    
    for src_file in src_files:
        if Path(src_file).exists():
            dst_file = project_dir / 'src' / Path(src_file).name
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {Path(src_file).name}")
    
    # 复制 common/utils.py
    if Path('src/common/utils.py').exists():
        common_dir = project_dir / 'src' / 'common'
        common_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2('src/common/utils.py', common_dir / 'utils.py')
        (common_dir / '__init__.py').write_text('', encoding='utf-8')
        print(f"  复制: common/utils.py")
    
    # 2. 复制数据文件
    print("\n2. 复制数据文件...")
    data_files = [
        'data/raw/束位数据.csv',
        'data/raw/束位数据.xlsx',
        'data/raw/束位监测数据.csv',
        'data/raw/束位监测数据.xlsx',
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            dst_file = project_dir / 'data' / 'raw' / Path(data_file).name
            shutil.copy2(data_file, dst_file)
            print(f"  复制: {Path(data_file).name}")
    
    # 3. 复制处理后的数据
    print("\n3. 复制处理后的数据...")
    processed_files = [
        'data/processed/X_train_static_random.csv',
        'data/processed/X_test_static_random.csv',
        'data/processed/y_train_static_random.csv',
        'data/processed/y_test_static_random.csv',
        'data/processed/X_train_static.csv',
        'data/processed/X_test_static.csv',
        'data/processed/y_train_static.csv',
        'data/processed/y_test_static.csv',
        'data/processed/X_train.csv',
        'data/processed/X_test.csv',
        'data/processed/y_train.csv',
        'data/processed/y_test.csv',
    ]
    
    for proc_file in processed_files:
        if Path(proc_file).exists():
            dst_file = project_dir / 'data' / 'processed' / Path(proc_file).name
            shutil.copy2(proc_file, dst_file)
            print(f"  复制: {Path(proc_file).name}")
    
    # 4. 复制结果文件
    print("\n4. 复制结果文件...")
    result_dirs = [
        'result/束位监测',
        'result/baseline_models',
    ]
    
    for result_dir_path in result_dirs:
        src_result = Path(result_dir_path)
        if src_result.exists():
            if 'baseline_models' in result_dir_path:
                dst_result = project_dir / 'result' / 'baseline_models'
            else:
                dst_result = project_dir / 'result'
                # 如果已经存在，需要合并
                if dst_result.exists() and src_result.name != 'baseline_models':
                    # 复制子目录
                    for item in src_result.iterdir():
                        dst_item = dst_result / item.name
                        if item.is_dir():
                            if dst_item.exists():
                                shutil.rmtree(dst_item)
                            shutil.copytree(item, dst_item)
                        else:
                            shutil.copy2(item, dst_item)
                    print(f"  合并结果目录: {src_result}")
                    continue
            
            if dst_result.exists():
                shutil.rmtree(dst_result)
            shutil.copytree(src_result, dst_result)
            print(f"  复制结果目录: {src_result}")
    
    # 5. 复制模型文件
    print("\n5. 复制模型文件...")
    model_files = [
        'models/pls_model.pkl',
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            dst_file = project_dir / 'models' / Path(model_file).name
            shutil.copy2(model_file, dst_file)
            print(f"  复制: {Path(model_file).name}")
    
    # 6. 创建 requirements.txt
    print("\n6. 创建 requirements.txt...")
    requirements = """jupyter>=1.1.1
lightgbm>=4.6.0
matplotlib>=3.10.7
numpy>=2.3.4
openpyxl>=3.1.5
pandas>=2.3.3
scikit-learn>=1.7.2
scipy>=1.16.2
seaborn>=0.13.2
tqdm>=4.67.1
xgboost>=3.1.1
"""
    (project_dir / 'requirements.txt').write_text(requirements, encoding='utf-8')
    print("  创建 requirements.txt")
    
    print(f"\n{project_name} 项目创建完成！")

def main():
    """主函数"""
    print("开始拆分项目为两个独立项目...")
    
    # 创建束流数据项目
    copy_beam_current_project()
    
    # 创建束位监测数据项目
    copy_beam_position_project()
    
    print(f"\n{'='*60}")
    print("项目拆分完成！")
    print(f"{'='*60}")
    print("\n下一步：")
    print("1. 更新两个项目中的文件路径引用")
    print("2. 为每个项目创建独立的 README.md")
    print("3. 测试两个项目是否能独立运行")

if __name__ == '__main__':
    main()
