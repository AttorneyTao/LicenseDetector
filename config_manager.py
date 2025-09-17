#!/usr/bin/env python3
"""
LLM配置管理工具

用于查看和切换LLM提供者配置的命令行工具。
"""

import argparse
import os
import sys
from core.config import LLM_CONFIG
from core.llm_provider import LLMProviderFactory, get_llm_provider


def show_current_config():
    """显示当前配置"""
    print("🔧 当前LLM配置:")
    print(f"   提供者: {LLM_CONFIG['provider']}")
    print(f"   Gemini模型: {LLM_CONFIG['gemini']['model']}")
    print(f"   Qwen模型: {LLM_CONFIG['qwen']['model']}")
    print(f"   Gemini API Key: {'✅ 已配置' if LLM_CONFIG['gemini']['api_key'] else '❌ 未配置'}")
    print(f"   Qwen API Key: {'✅ 已配置' if LLM_CONFIG['qwen']['api_key'] else '❌ 未配置'}")


def test_provider(provider_name):
    """测试指定提供者"""
    try:
        print(f"🧪 测试 {provider_name} 提供者...")
        provider = LLMProviderFactory.get_provider(provider_name)
        response = provider.generate("Hello, please respond with 'OK'")
        print(f"   ✅ 测试成功，响应长度: {len(response)}")
        return True
    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        return False


def switch_provider(provider_name):
    """切换提供者（修改.env文件）"""
    env_file = '.env'
    if not os.path.exists(env_file):
        print("❌ .env文件不存在")
        return False
    
    # 读取现有内容
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 更新或添加LLM_PROVIDER行
    found = False
    for i, line in enumerate(lines):
        if line.startswith('LLM_PROVIDER='):
            lines[i] = f'LLM_PROVIDER={provider_name}\n'
            found = True
            break
    
    if not found:
        lines.append(f'LLM_PROVIDER={provider_name}\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ 已将默认提供者切换为: {provider_name}")
    print("⚠️  请重新启动应用以使配置生效")
    return True


def main():
    parser = argparse.ArgumentParser(description='LLM配置管理工具')
    parser.add_argument('--show', '-s', action='store_true', help='显示当前配置')
    parser.add_argument('--test', '-t', choices=['gemini', 'qwen'], help='测试指定提供者')
    parser.add_argument('--switch', '-w', choices=['gemini', 'qwen'], help='切换默认提供者')
    parser.add_argument('--test-all', '-a', action='store_true', help='测试所有可用提供者')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_config()
    elif args.test:
        test_provider(args.test)
    elif args.switch:
        switch_provider(args.switch)
    elif args.test_all:
        show_current_config()
        print("\n🧪 测试所有提供者:")
        test_provider('gemini')
        test_provider('qwen')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()