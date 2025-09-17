#!/usr/bin/env python3
"""检查最新输出结果中的license_files字段"""

import pandas as pd
import sys

def check_latest_output():
    try:
        # 读取最新的输出文件
        df = pd.read_excel('./outputs/output_latest.xlsx')
        
        print("📊 输出文件检查结果:")
        print(f"总行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        
        if 'license_files' in df.columns:
            print("\n🔍 license_files 字段内容:")
            for idx, row in df.iterrows():
                repo_url = row.get('repository_url', 'N/A')
                license_files = row.get('license_files', 'N/A')
                print(f"仓库: {repo_url}")
                print(f"License Files: {license_files}")
                print("-" * 50)
        else:
            print("❌ 未找到 license_files 字段")
            
        # 检查其他相关字段
        if 'concluded_license' in df.columns:
            print("\n📋 concluded_license 字段内容:")
            for idx, row in df.iterrows():
                concluded_license = row.get('concluded_license', 'N/A')
                print(f"Concluded License: {concluded_license}")
                
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    check_latest_output()