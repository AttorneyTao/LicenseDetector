#!/usr/bin/env python3
"""
测试PyPI流程中GitHub优先级处理
"""

import asyncio
import logging
from core.pypi_utils import process_pypi_repository

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pypi_with_github():
    """测试具有GitHub仓库的PyPI包"""
    test_url = "https://pypi.org/project/requests/"
    
    logger.info(f"Testing PyPI package with GitHub repository: {test_url}")
    
    result = await process_pypi_repository(test_url)
    
    logger.info("=== Test Results ===")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Component Name: {result.get('component_name')}")
    logger.info(f"Repo URL: {result.get('repo_url')}")
    logger.info(f"License Files: {result.get('license_files')}")
    logger.info(f"License Type: {result.get('license_type')}")
    logger.info(f"License Determination Reason: {result.get('license_determination_reason')}")
    logger.info(f"Has GitHub Analysis: {result.get('license_analysis') is not None}")
    
    # 验证GitHub优先级
    if result.get('repo_url') and 'github.com' in result.get('repo_url', ''):
        if result.get('license_determination_reason') == "Analyzed via GitHub repository (primary source)":
            logger.info("✅ PASS: GitHub analysis used as primary source")
        else:
            logger.warning("❌ FAIL: GitHub should be used as primary source")
            
        if result.get('license_files') and 'github.com' in result.get('license_files', ''):
            logger.info("✅ PASS: License files point to GitHub")
        else:
            logger.warning("❌ FAIL: License files should point to GitHub when GitHub repo is available")
    else:
        logger.info("ℹ️ No GitHub repository found, using PyPI analysis")
    
    return result

async def test_pypi_without_github():
    """测试没有GitHub仓库的PyPI包"""
    test_url = "https://pypi.org/project/six/"  # 这个包通常没有GitHub链接
    
    logger.info(f"Testing PyPI package without GitHub repository: {test_url}")
    
    result = await process_pypi_repository(test_url)
    
    logger.info("=== Test Results ===")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Component Name: {result.get('component_name')}")
    logger.info(f"Repo URL: {result.get('repo_url')}")
    logger.info(f"License Files: {result.get('license_files')}")
    logger.info(f"License Type: {result.get('license_type')}")
    logger.info(f"License Determination Reason: {result.get('license_determination_reason')}")
    
    # 验证PyPI fallback
    if not result.get('repo_url') or 'github.com' not in result.get('repo_url', ''):
        if result.get('license_determination_reason') == "Fetched from PyPI registry":
            logger.info("✅ PASS: PyPI analysis used when no GitHub repo")
        else:
            logger.warning("❌ FAIL: Should use PyPI analysis when no GitHub repo")
            
        if 'pypi.org' in result.get('license_files', ''):
            logger.info("✅ PASS: License files point to PyPI")
        else:
            logger.warning("❌ FAIL: License files should point to PyPI when no GitHub repo")
    
    return result

async def main():
    """运行所有测试"""
    logger.info("Starting PyPI GitHub priority tests...")
    
    try:
        logger.info("\n" + "="*50)
        logger.info("Test 1: PyPI package with GitHub repository")
        logger.info("="*50)
        await test_pypi_with_github()
        
        logger.info("\n" + "="*50)
        logger.info("Test 2: PyPI package without GitHub repository")
        logger.info("="*50)
        await test_pypi_without_github()
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())