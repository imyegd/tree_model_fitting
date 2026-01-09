#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为两个项目创建必要的 __init__.py 文件
"""

from pathlib import Path

# 为 beam_position 项目创建 __init__.py
beam_position_src = Path('beam_position/src')
beam_position_common = beam_position_src / 'common'

# 创建 src/__init__.py
(beam_position_src / '__init__.py').write_text('"""束位监测数据处理模块"""\n', encoding='utf-8')

# 确保 common/__init__.py 存在
if beam_position_common.exists():
    (beam_position_common / '__init__.py').write_text('"""共享工具模块"""\n', encoding='utf-8')

# 为 beam_current 项目创建 __init__.py
beam_current_src = Path('beam_current/src')
(beam_current_src / '__init__.py').write_text('"""束流数据处理模块"""\n', encoding='utf-8')

print("创建 __init__.py 文件完成！")
