#!/usr/bin/env python3
"""
测试许可证分析功能的脚本

用于测试改进后的许可证分析提示词在复杂许可证文本上的表现。
"""

from core.utils import analyze_license_content
from core.llm_provider import get_llm_provider
import json


def test_hadoop_license():
    """测试Apache Hadoop许可证文本"""
    
    # Apache Hadoop的许可证文本（简化版，包含主要部分）
    hadoop_license_text = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

   [Apache License text continues...]

   END OF TERMS AND CONDITIONS

APACHE HADOOP SUBCOMPONENTS:

The Apache Hadoop project contains subcomponents with separate copyright
notices and license terms. Your use of the source code for the these
subcomponents is subject to the terms and conditions of the following
licenses.

For the org.apache.hadoop.util.bloom.* classes:
Copyright (c) 2005, European Commission project OneLab under contract
034819 (http://www.one-lab.org)
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
[BSD-3-Clause license text]

For portions of the native implementation of slicing-by-8 CRC calculation:
Copyright 2008,2009,2010 Massachusetts Institute of Technology.
All rights reserved. Use of this source code is governed by a
BSD-style license that can be found in the LICENSE file.

For src/main/native/src/org/apache/hadoop/io/compress/lz4/{lz4.h,lz4.c,lz4hc.h,lz4hc.c},
Copyright (C) 2011-2014, Yann Collet.
BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

The binary distribution of this product bundles binaries of leveldbjni
Copyright (c) 2011 FuseSource Corp. All rights reserved.
[BSD-3-Clause license text]

The binary distribution of this product bundles binaries of leveldb
Copyright (c) 2011 The LevelDB Authors. All rights reserved.
[BSD-3-Clause license text]

The binary distribution of this product bundles binaries of snappy
Copyright 2011, Google Inc.
All rights reserved.
[BSD-3-Clause license text]

For bootstrap-3.0.2 and related dependencies:
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy
[MIT license text]

For jQuery and related dependencies:
Copyright jQuery Foundation and other contributors, https://jquery.org/
[MIT license text]

The binary distribution of this product bundles these dependencies under the
following license:
servlet-api 2.5, jsp-api 2.1, Streaming API for XML 1.0
COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0

The binary distribution of this product bundles these dependencies under the
following license:
Jersey 1.9, JAXB API bundle for GlassFish V3 2.2.2, JAXB RI 2.2.3
COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL)Version 1.1

The binary distribution of this product bundles these dependencies under the
following license:
JUnit 4.11, ecj-4.3.1.jar
Eclipse Public License - v 1.0

The binary distribution of this product bundles these dependencies under the
following license:
Protocol Buffer Java API 2.5.0
Copyright 2014, Google Inc.  All rights reserved.
[BSD-3-Clause license text]
"""
    
    print("🧪 测试Apache Hadoop许可证分析...")
    print("📄 许可证文本长度:", len(hadoop_license_text), "字符")
    
    try:
        result = analyze_license_content(hadoop_license_text)
        
        if result:
            print("\n✅ 分析成功!")
            print("📋 分析结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查关键字段
            if "spdx_expression" in result:
                print(f"\n🎯 SPDX表达式: {result['spdx_expression']}")
            else:
                print(f"\n🎯 主要许可证: {result.get('main_licenses', result.get('licenses', 'Unknown'))}")
                
            if result.get('has_third_party_licenses'):
                print(f"📦 包含第三方许可证: 是")
                if "bundled_licenses" in result:
                    print(f"📦 捆绑的许可证: {result['bundled_licenses']}")
            else:
                print(f"📦 包含第三方许可证: 否")
                
        else:
            print("❌ 分析失败: 返回结果为空")
            
    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")


def test_simple_mit_license():
    """测试简单的MIT许可证"""
    mit_text = """
MIT License

Copyright (c) 2023 Example Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    print("\n\n🧪 测试简单MIT许可证分析...")
    
    try:
        result = analyze_license_content(mit_text)
        
        if result:
            print("✅ 分析成功!")
            print("📋 分析结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ 分析失败: 返回结果为空")
            
    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")


def test_llm_provider():
    """测试LLM提供者"""
    print("🔧 当前LLM提供者测试...")
    
    try:
        provider = get_llm_provider()
        print(f"✅ LLM提供者类型: {type(provider).__name__}")
        
        # 简单测试
        response = provider.generate("请用中文回答：1+1等于几？")
        print(f"🔍 简单测试响应: {response[:50]}...")
        
    except Exception as e:
        print(f"❌ LLM提供者测试失败: {str(e)}")


if __name__ == "__main__":
    # 测试LLM提供者
    test_llm_provider()
    
    # 测试简单许可证
    test_simple_mit_license()
    
    # 测试复杂许可证
    test_hadoop_license()