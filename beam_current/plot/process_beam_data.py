#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
束流数据预处理：筛选 8月31日 2:40 之前的数据，保存为 beamdata.csv
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', '束流.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'beamdata.csv')


def process_beam_data():
    """筛选 0831 两点四十之前的数据并保存"""
    print(f"正在读取: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    df['时间'] = pd.to_datetime(df['时间'])
    cutoff = pd.Timestamp('2025-08-31 02:49:00')
    df_filtered = df[df['时间'] < cutoff].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_filtered.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    print(f"原始数据: {len(df)} 条")
    print(f"筛选后: {len(df_filtered)} 条 (8月31日 2:40 之前)")
    print(f"已保存: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == '__main__':
    process_beam_data()
