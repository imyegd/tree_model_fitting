"""
列出所有字体
"""
from matplotlib import pyplot as plt
import matplotlib
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

for i in a:
    print(i)

"""
测试字体支持情况
"""

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# import matplotlib.font_manager as fm

# def test_font_support():
#     """测试字体对中英文的支持情况"""
    
#     # 要测试的字体
#     fonts_to_test = [
#         'Droid Sans Fallback',
#         'Noto Sans CJK JP', 
#         'Noto Serif CJK JP',
#         'Liberation Sans',
#         'DejaVu Sans'
#     ]
    
#     # 测试文本
#     test_texts = {
#         'English': 'Hello World 123',
#         'Chinese': '你好世界 真实值 预测值',
#         'Numbers': '0.123456789'
#     }
    
#     print("字体支持测试结果：")
#     print("=" * 50)
    
#     for font_name in fonts_to_test:
#         print(f"\n测试字体: {font_name}")
#         print("-" * 30)
        
#         try:
#             # 创建字体属性
#             font_prop = FontProperties(family=font_name)
            
#             # 测试每种文本
#             for text_type, text in test_texts.items():
#                 try:
#                     # 创建测试图形
#                     fig, ax = plt.subplots(figsize=(2, 1))
#                     ax.text(0.5, 0.5, text, fontproperties=font_prop, ha='center', va='center')
#                     ax.set_xlim(0, 1)
#                     ax.set_ylim(0, 1)
#                     ax.axis('off')
                    
#                     # 保存测试图片
#                     test_filename = f"font_test_{font_name.replace(' ', '_')}_{text_type}.png"
#                     plt.savefig(test_filename, dpi=100, bbox_inches='tight')
#                     plt.close()
                    
#                     print(f"  {text_type}: ✓ 支持")
                    
#                 except Exception as e:
#                     print(f"  {text_type}: ✗ 不支持 - {str(e)}")
                    
#         except Exception as e:
#             print(f"  字体加载失败: {str(e)}")

# def find_best_font():
#     """找到最佳的中英文字体组合"""
    
#     print("\n推荐的中英文字体设置：")
#     print("=" * 50)
    
#     # 方案1：使用Noto字体
#     print("方案1 - 使用Noto字体：")
#     print("matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']")
    
#     # 方案2：使用Liberation Sans + 手动设置中文
#     print("\n方案2 - 使用Liberation Sans + 手动设置中文：")
#     print("matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']")
#     print("# 然后在绘图时手动设置中文字体")
    
#     # 方案3：混合字体
#     print("\n方案3 - 混合字体设置：")
#     print("matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'Noto Sans CJK JP', 'DejaVu Sans']")

# if __name__ == "__main__":
#     test_font_support()
#     find_best_font()