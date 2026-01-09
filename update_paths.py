#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新两个项目中的文件路径引用
"""

import re
from pathlib import Path

def update_beam_current_paths():
    """更新束流数据项目中的路径"""
    project_dir = Path('beam_current')
    src_dir = project_dir / 'src'
    
    print("更新束流数据项目中的路径...")
    
    # 需要更新的文件
    files_to_update = [
        'linear.py',
        'tree_model.py',
        'mlp.py',
        'LSTM.py',
        'PLS.py',
        'data_process.py',
    ]
    
    for filename in files_to_update:
        filepath = src_dir / filename
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # 更新数据路径
        content = re.sub(r'\./data/束流\.csv', './data/raw/束流.csv', content)
        content = re.sub(r'\./data/split_data\.npz', './data/processed/split_data.npz', content)
        content = re.sub(r'\./data/scaler\.pkl', './data/processed/scaler.pkl', content)
        content = re.sub(r'\./data/束流', './data/raw/束流', content)
        
        # 更新结果路径（确保使用相对路径）
        content = re.sub(r'\./result/束流/', './result/', content)
        content = re.sub(r'f"\./result/束流/', 'f"./result/', content)
        
        # 更新模型路径
        content = re.sub(r'\./models/', './models/', content)
        
        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  更新: {filename}")

def update_beam_position_paths():
    """更新束位监测数据项目中的路径"""
    project_dir = Path('beam_position')
    src_dir = project_dir / 'src'
    
    print("\n更新束位监测数据项目中的路径...")
    
    # 需要更新的文件
    files_to_update = [
        '01_data_prep.py',
        '02_baseline_models.py',
        'position.py',
        'plot.py',
        'XGboost.py',
        'PLS_test.py',
    ]
    
    for filename in files_to_update:
        filepath = src_dir / filename
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # 更新数据路径
        content = re.sub(r'data\\束位', 'data/raw/束位', content)
        content = re.sub(r'data/束位', 'data/raw/束位', content)
        content = re.sub(r'\./data/束位', './data/raw/束位', content)
        content = re.sub(r'data\\束位监测', 'data/raw/束位监测', content)
        content = re.sub(r'data/束位监测', 'data/raw/束位监测', content)
        content = re.sub(r'\./data/束位监测', './data/raw/束位监测', content)
        
        # 更新处理后的数据路径
        content = re.sub(r'data/processed/', './data/processed/', content)
        content = re.sub(r'PROCESSED_DATA_PATH = \'data/processed\'', 
                        'PROCESSED_DATA_PATH = \'./data/processed\'', content)
        content = re.sub(r'RAW_DATA_PATH = \'data/raw\'', 
                        'RAW_DATA_PATH = \'./data/raw\'', content)
        
        # 更新结果路径
        content = re.sub(r'\./result/束位监测/', './result/', content)
        content = re.sub(r'result/束位监测/', 'result/', content)
        content = re.sub(r'result/baseline_models', './result/baseline_models', content)
        content = re.sub(r'\'result/baseline_models\'', '\'./result/baseline_models\'', content)
        
        # 更新模型路径
        content = re.sub(r'\./models/', './models/', content)
        content = re.sub(r'\./data/processed/split_data\.npz', './data/processed/split_data.npz', content)
        content = re.sub(r'\./models/pls_model\.pkl', './models/pls_model.pkl', content)
        
        # 更新导入路径
        content = re.sub(r'sys\.path\.append\(str\(Path\(__file__\)\.parent\.parent\)\)',
                        'sys.path.append(str(Path(__file__).parent.parent))', content)
        content = re.sub(r'from common\.utils import', 'from src.common.utils import', content)
        
        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  更新: {filename}")
    
    # 更新 01_data_prep.py 中的导入
    prep_file = src_dir / '01_data_prep.py'
    if prep_file.exists():
        content = prep_file.read_text(encoding='utf-8')
        # 修复导入路径
        content = re.sub(
            r'import sys\s+from pathlib import Path\s+sys\.path\.append\(str\(Path\(__file__\)\.parent\.parent\)\)\s+from common\.utils import',
            'from src.common.utils import',
            content,
            flags=re.MULTILINE
        )
        # 如果上面的替换没成功，尝试更简单的方式
        if 'from common.utils import' in content or 'from src.common.utils import' not in content:
            content = content.replace('from common.utils import', 'from src.common.utils import')
            content = content.replace(
                'import sys\nfrom pathlib import Path\nsys.path.append(str(Path(__file__).parent.parent))',
                ''
            )
        prep_file.write_text(content, encoding='utf-8')
        print(f"  更新导入路径: 01_data_prep.py")
    
    # 更新 02_baseline_models.py 中的导入
    baseline_file = src_dir / '02_baseline_models.py'
    if baseline_file.exists():
        content = baseline_file.read_text(encoding='utf-8')
        content = content.replace('from common.utils import', 'from src.common.utils import')
        content = content.replace(
            'import sys\nfrom pathlib import Path\nsys.path.append(str(Path(__file__).parent.parent))',
            ''
        )
        baseline_file.write_text(content, encoding='utf-8')
        print(f"  更新导入路径: 02_baseline_models.py")

def main():
    """主函数"""
    print("开始更新项目路径...")
    update_beam_current_paths()
    update_beam_position_paths()
    print("\n路径更新完成！")

if __name__ == '__main__':
    main()
