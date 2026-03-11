#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 crates.io 集成是否正常工作
Quick verification script for crates.io integration
"""

import sys
import importlib.util

def check_imports():
    """检查必要的模块是否可以导入"""
    print("=" * 60)
    print("检查模块导入...")
    print("=" * 60)
    
    modules_to_check = [
        "core.crate_utils",
        "core.npm_utils",
        "aiohttp",
        "aiofiles",
        "bs4",
        "packaging",
    ]
    
    all_ok = True
    for module in modules_to_check:
        try:
            if "." in module:
                # 处理子模块
                parts = module.split(".")
                mod = importlib.import_module(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)
            else:
                importlib.import_module(module)
            print(f"✓ {module:30s} - OK")
        except Exception as e:
            print(f"✗ {module:30s} - FAILED: {e}")
            all_ok = False
    
    return all_ok


def check_main_integration():
    """检查 main.py 中的集成"""
    print("\n" + "=" * 60)
    print("检查 main.py 集成...")
    print("=" * 60)
    
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("from core.crate_utils import", "导入语句"),
            ("from core.npm_utils import", "npm_utils 导入"),
            ("is_crate_pkg =", "crate 检测逻辑"),
            ("is_npm_pkg =", "npm 检测逻辑"),
            ("process_crate_repository", "调用处理函数"),
            ("process_npm_repository", "调用 npm 处理函数"),
        ]
        
        all_ok = True
        for pattern, description in checks:
            if pattern in content:
                print(f"✓ {description:20s} - OK")
            else:
                print(f"✗ {description:20s} - NOT FOUND")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print(f"✗ 检查失败：{e}")
        return False


def check_crate_utils_structure():
    """检查 crate_utils.py 的结构"""
    print("\n" + "=" * 60)
    print("检查 crate_utils.py 结构...")
    print("=" * 60)
    
    try:
        from core import crate_utils
        
        required_functions = [
            "process_crate_repository",
            "resolve_crate_version",
            "_parse_crate_name",
            "_fetch_crate_info",
            "_fetch_version_info",
            "_list_all_versions",
        ]
        
        all_ok = True
        for func_name in required_functions:
            if hasattr(crate_utils, func_name):
                print(f"✓ {func_name:30s} - OK")
            else:
                print(f"✗ {func_name:30s} - NOT FOUND")
                all_ok = False
        
        # 检查常量
        constants = [
            "CRATES_IO_API_BASE",
            "CRATES_IO_BASE",
            "CrateAPIError",
        ]
        
        for const in constants:
            if hasattr(crate_utils, const):
                print(f"✓ {const:30s} - OK")
            else:
                print(f"✗ {const:30s} - NOT FOUND")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print(f"✗ 检查失败：{e}")
        return False


def quick_api_test():
    """快速测试 API（可选）"""
    print("\n" + "=" * 60)
    print("快速 API 测试（跳过网络请求）...")
    print("=" * 60)
    
    try:
        from core.crate_utils import _parse_crate_name, _normalize_requested_crate_version
        
        # 测试 URL 解析
        test_cases = [
            ("serde", "serde"),
            ("https://crates.io/crates/tokio", "tokio"),
            ("crates.io/crates/reqwest", "reqwest"),
            ("https://crates.io/crates/serde/1.0.0", "serde"),
        ]
        
        all_ok = True
        for input_val, expected in test_cases:
            result = _parse_crate_name(input_val)
            if result == expected:
                print(f"✓ _parse_crate_name('{input_val}') = '{result}' - OK")
            else:
                print(f"✗ _parse_crate_name('{input_val}') = '{result}', expected '{expected}' - FAILED")
                all_ok = False
        
        # 测试版本标准化
        version_tests = [
            (None, None),
            ("", None),
            ("1.0.0", "1.0.0"),
            ("v1.0.0", "1.0.0"),
            ("V2.0.0", "2.0.0"),
        ]
        
        for input_val, expected in version_tests:
            result = _normalize_requested_crate_version(input_val)
            if result == expected:
                print(f"✓ _normalize_requested_crate_version({input_val!r}) = {result!r} - OK")
            else:
                print(f"✗ _normalize_requested_crate_version({input_val!r}) = {result!r}, expected {expected!r} - FAILED")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        return False


def main():
    """主验证函数"""
    print("\n" + "=" * 60)
    print("Crates.io 集成验证工具")
    print("=" * 60 + "\n")
    
    results = []
    
    # 1. 检查导入
    results.append(("模块导入", check_imports()))
    
    # 2. 检查 main.py 集成
    results.append(("main.py 集成", check_main_integration()))
    
    # 3. 检查 crate_utils 结构
    results.append(("crate_utils 结构", check_crate_utils_structure()))
    
    # 4. 快速功能测试
    results.append(("功能测试", quick_api_test()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有验证通过！集成完成。")
        print("\n下一步:")
        print("1. 运行 python test_crate.py 进行完整测试")
        print("2. 在 input.xlsx 中添加 crates.io 包")
        print("3. 运行 python main.py 开始分析")
    else:
        print("⚠️  部分验证未通过，请检查错误信息")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
