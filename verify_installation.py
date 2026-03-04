#!/usr/bin/env python
"""
项目完整性验证脚本
验证所有新增和修改的文件是否正常工作
"""

import os
import sys
import json
from pathlib import Path


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (未找到)")
        return False


def check_python_files():
    """检查Python文件是否有效"""
    print("\n📝 检查Python文件...")
    files_to_check = [
        ("main.py", "主程序"),
        ("api.py", "API应用"),
        ("core/email_utils.py", "邮件工具"),
        ("test_api.py", "测试脚本"),
    ]
    
    all_valid = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_valid = False
        else:
            # 检查语法
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    compile(f.read(), filepath, 'exec')
            except Exception as e:
                print(f"  ⚠ 语法错误: {e}")
                all_valid = False
    
    return all_valid


def check_documentation():
    """检查文档文件"""
    print("\n📚 检查文档文件...")
    docs = [
        ("API_USAGE.md", "API使用指南"),
        ("CHANGES.md", "改动说明"),
        ("QUICKSTART.md", "快速开始"),
        (".env.example", "配置模板"),
    ]
    
    all_exist = True
    for filepath, description in docs:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_dependencies():
    """检查依赖是否安装"""
    print("\n📦 检查Python依赖...")
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('multipart', 'python-multipart'),
        ('pandas', 'Pandas'),
        ('requests', 'Requests'),
    ]
    
    all_installed = True
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (未安装)")
            all_installed = False
    
    return all_installed


def check_main_py_structure():
    """检查main.py是否包含必要的修改"""
    print("\n🔍 检查main.py的修改...")
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    checks = [
        ("import argparse", "argparse导入"),
        ("--api", "API参数"),
        ("run_api_server", "API函数调用"),
        ("from api import run_api_server", "API导入"),
    ]
    
    all_found = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} (未找到)")
            all_found = False
    
    return all_found


def check_api_py_structure():
    """检查api.py的完整性"""
    print("\n🔍 检查api.py的结构...")
    
    with open("api.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    checks = [
        ("/api/v1/analyze", "分析并邮件端点"),
        ("/api/v1/analyze-and-download", "分析下载端点"),
        ("/health", "健康检查端点"),
        ("process_github_repository", "处理仓库函数"),
        ("send_analysis_result", "发送结果函数"),
    ]
    
    all_found = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} (未找到)")
            all_found = False
    
    return all_found


def check_pyproject_toml():
    """检查pyproject.toml的依赖"""
    print("\n🔍 检查pyproject.toml的依赖...")
    
    with open("pyproject.toml", "r", encoding="utf-8") as f:
        content = f.read()
    
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("python-multipart", "python-multipart"),
    ]
    
    all_found = True
    for package, name in packages:
        if package in content:
            print(f"✓ {name} 已添加到dependencies")
        else:
            print(f"✗ {name} 未找到 (需要手动添加)")
            all_found = False
    
    return all_found


def check_directory_structure():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = [
        ("core", "核心模块"),
        ("outputs", "输出目录"),
        ("logs", "日志目录"),
        ("temp", "临时目录"),
    ]
    
    all_exist = True
    for dirname, description in required_dirs:
        if os.path.isdir(dirname):
            print(f"✓ {description}: {dirname}/")
        else:
            print(f"⚠ {description}: {dirname}/ (不存在，运行时自动创建)")
    
    return True


def main():
    """运行所有检查"""
    print("=" * 60)
    print("GitHub License Analyzer - 项目完整性验证")
    print("=" * 60)
    
    results = {
        "Python文件": check_python_files(),
        "文档文件": check_documentation(),
        "Python依赖": check_dependencies(),
        "main.py修改": check_main_py_structure(),
        "api.py结构": check_api_py_structure(),
        "pyproject.toml": check_pyproject_toml(),
        "目录结构": check_directory_structure(),
    }
    
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    for check_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{check_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有项目均通过验证！")
        print("\n🚀 接下来可以：")
        print("  1. CLI模式: uv run main.py")
        print("  2. API模式: python main.py --api")
        print("  3. 测试API: python test_api.py")
        print("\n📚 查看文档:")
        print("  - QUICKSTART.md - 快速开始")
        print("  - API_USAGE.md - 详细API文档")
        print("  - CHANGES.md - 改动说明")
        return 0
    else:
        print("✗ 验证过程中发现问题")
        print("\n请执行以下命令修复:")
        print("  uv sync  # 重新安装依赖")
        return 1


if __name__ == "__main__":
    sys.exit(main())
