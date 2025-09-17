#!/usr/bin/env python3
"""
测试完整的Maven仓库分析流程

测试hadoop-client的完整分析流程，包括许可证URL获取和分析。
"""

import asyncio
from core.maven_utils import get_license_from_maven_url
import json


async def test_hadoop_client():
    """测试Apache Hadoop Client的完整流程"""
    
    test_url = "https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-client"
    
    print("🧪 测试Maven URL:", test_url)
    print("⏳ 开始分析...")
    
    try:
        result = await get_license_from_maven_url(test_url)
        
        if result:
            print("✅ 分析成功!")
            print("📋 完整结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查关键信息
            if "concluded_license" in result:
                print(f"\n🎯 结论性许可证: {result['concluded_license']}")
            if "license_files" in result:
                print(f"📄 许可证文件: {result['license_files']}")
            if "copyright_notice" in result:
                print(f"©️ 版权声明: {result['copyright_notice']}")
                
        else:
            print("❌ 分析失败: 返回结果为空")
            
    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_hadoop_client())