# ============================================================================
# Configuration and Setup Section
# ============================================================================
# This section handles all the initial setup including:
# - Importing required libraries for HTTP requests, JSON processing, and data analysis
# - Setting up logging configuration with multiple handlers for different aspects
# - Loading environment variables for API keys and configuration
# - Configuring API settings for GitHub and Gemini LLM
# - Setting up UTF-8 encoding for proper character handling

# Set the default encoding for stdout and stderr to utf-8
import sys
import codecs
import re




# ============================================================================
# Import Required Libraries Section
# ============================================================================
import os
import time
import logging
from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import unquote

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from tqdm import tqdm

#=============================================================================
# Import internal packages
#=============================================================================
from core.logging_utils import setup_logging
from core.github_utils import GitHubAPI
from core.config import LLM_CONFIG, SCORE_THRESHOLD, MAX_CONCURRENCY, RESULT_COLUMNS_ORDER
from core.utils import get_concluded_license, extract_thirdparty_dirs_column
from core.go_utils import  get_github_url_from_pkggo

# ============================================================================
# Load Prompts Section
import yaml

from core.github_utils import process_github_repository
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# ============================================================================  
# Configuration and Setup Section
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)


# Set up logging
setup_logging()

load_dotenv()

# Configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
logger = logging.getLogger('main')  # Use the main logger for general application logging
substep_logger =logging.getLogger('substep')
url_logger = logging.getLogger('url_construction')

logger.info(f"LLM analysis is {'enabled' if USE_LLM else 'disabled'}")


sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)



# Validate LLM API configuration only if LLM is enabled
if USE_LLM:
    # Get the current provider from LLM_CONFIG
    provider = LLM_CONFIG.get("provider", "gemini")
    
    if provider.lower() == "gemini":
        gemini_config = LLM_CONFIG.get("gemini", {})
        api_key = gemini_config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Initialize Gemini client using the proper way according to the llm_provider abstraction
        # Remove direct genai.configure call since it's handled in llm_provider.py
        model = gemini_config.get("model", "gemini-2.5-flash")
        logger.info(f"Initialized Gemini API with model: {model}")
    elif provider.lower() == "qwen":
        qwen_config = LLM_CONFIG.get("qwen", {})
        api_key = qwen_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("DASHSCOPE_API_KEY environment variable not set")
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        # Qwen uses OpenAI-compatible API, no specific initialization needed here
        model = qwen_config.get("model", "qwen-plus")
        logger.info(f"Initialized Qwen API with model: {model}")
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")





# ============================================================================
# License Analysis Functions
# ============================================================================
# Implements functions for:
# - Analyzing license content using LLM
# - Extracting copyright information
# - Constructing copyright notices
# - Finding GitHub URLs from package URLs
# - Handling different license formats and types

# ============================================================================
# Main Execution Function
# ============================================================================
# The main entry point that:
# - Initializes the application
# - Reads input data
# - Processes repositories
# - Generates output
# - Handles errors and cleanup

import asyncio
from tqdm import tqdm

async def process_all_repos(api, df, max_concurrency=MAX_CONCURRENCY):
    logger = logging.getLogger('main')
    sem = asyncio.Semaphore(max_concurrency)
    results = {}
    last_save_time = time.time()
    SAVE_INTERVAL = 30  # 每30秒保存一次临时文件
    
    # 添加一个计数器来跟踪当前运行的任务数
    running_tasks = 0
    
    async def save_temp_results():
        """保存当前已处理的结果到临时文件"""
        try:
            # 只保存已完成的结果
            current_results = [results[i] for i in range(len(df)) if i in results]
            temp_df = pd.DataFrame(current_results)
            
            # 使用固定文件名，这样总是覆盖最新的临时结果
            temp_file = "temp/temp_results_latest.csv"
            os.makedirs("temp", exist_ok=True)
            temp_df.to_csv(temp_file, index=False, encoding='utf-8')
            
            # 同时保存一个带时间戳的备份
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"temp/temp_results_{timestamp}.csv"
            temp_df.to_csv(backup_file, index=False, encoding='utf-8')
            
            logger.info(f"已保存中间结果，完成度: {len(current_results)}/{len(df)}")
        except Exception as e:
            logger.error(f"保存临时文件失败: {str(e)}", exc_info=True)

    async def process_with_progress(pbar):
        nonlocal last_save_time, running_tasks

        async def sem_task(row, index):
            nonlocal last_save_time, running_tasks
            async with sem:
                name = None  # 保证异常时也能访问 name
                try:
                    running_tasks += 1
                    logger.info(f"当前并发任务数: {running_tasks}")

                    from core.github_utils import normalize_github_url
                    from core.maven_utils import analyze_maven_repository_url
                    original_url = row["github_url"]  # 保存原始URL
                    url = normalize_github_url(original_url)
                    version = row.get("version")
                    name = row.get("name", None)

                    # 新增：判断是否为 Go 包
                    is_go_pkg = False
                    if isinstance(url, str):
                        # 支持 pkg.go.dev、go.dev、go module path 等常见格式
                        if url.startswith("https://pkg.go.dev/") or url.startswith("https://go.dev/") or re.match(r"^go\.[\w\.-]+/", url):
                            is_go_pkg = True

                    if is_go_pkg:
                        logger.info(f"检测到 Go 包 URL: {url}，尝试 get_github_url")
                        github_info = await get_github_url_from_pkggo(url, version, name)
                        github_url = github_info.get("github_url")
                        if github_url:
                            logger.info(f"get_github_url 成功，继续走 GitHub 流程: {github_url}")
                            result = await process_github_repository(
                                api,
                                github_url,
                                version,
                                name=name
                            )
                        else:
                            logger.info(f"get_github_url 失败,改用大模型方案")
                            result = await process_github_repository(
                                api,
                                url,
                                version,
                                name=name
                            )
                    else:
                        # 修改：对于 Maven URL（包括 mvnrepository.com 和 repo1.maven.org），先走默认的 GitHub 流程
                        is_maven_url = isinstance(url, str) and (
                            "mvnrepository.com/artifact" in url or 
                            "repo1.maven.org/maven2" in url
                        )
                        if is_maven_url:
                            logger.info(f"检测到 Maven URL: {url}，先尝试 GitHub 流程")
                            result = await process_github_repository(
                                api,
                                url,
                                version,
                                name=name
                            )
                            
                            # 如果 GitHub 流程不成功，再调用 Maven 处理函数
                            if result.get("status") != "success":
                                logger.info(f"GitHub 流程未成功，调用 Maven 处理函数")
                                try:
                                    from core.maven_utils import analyze_maven_repository_url
                                    maven_result = analyze_maven_repository_url(url)
                                    
                                    # 转换 Maven 结果为标准格式
                                    license_file_url = f"https://mvnrepository.com/artifact/{maven_result['group_id']}/{maven_result['artifact_id']}/{maven_result.get('version', '')}"
                                    
                                    # 修复：正确处理copyright信息
                                    copyright_notice = maven_result.get('copyright')
                                    if not copyright_notice:
                                        # 如果没有从Maven结果中获取到copyright，构造一个默认的
                                        org_parts = maven_result['group_id'].split(".")
                                        orgname = org_parts[1] if len(org_parts) > 1 else org_parts[0]
                                        copyright_notice = f"Copyright (c) {datetime.now().year} {orgname.capitalize()}"
                                    
                                    result = {
                                        "input_url": original_url,  # 使用原始URL
                                        "repo_url": None,
                                        "input_version": version,
                                        "resolved_version": maven_result.get('version'),
                                        "used_default_branch": False,
                                        "component_name": name or maven_result['artifact_id'],
                                        "license_files": license_file_url,
                                        "license_analysis": {
                                            "license_determination_reason": "Fetched from Maven Central POM",
                                            "license_source": maven_result.get('license_source', 'maven_central')
                                        },
                                        "license_type": maven_result.get('license'),
                                        "has_license_conflict": False,
                                        "readme_license": None,
                                        "license_file_license": maven_result.get('license'),
                                        "copyright_notice": copyright_notice,  # 修复：正确使用从Maven结果中获取的copyright
                                        "status": "success",
                                        "input_name": name,
                                    }
                                except Exception as e:
                                    logger.warning(f"Maven 处理失败: {e}")
                                    # 保持原来的错误结果
                        else:
                            # 非 Maven URL，直接走原来的 GitHub 流程
                            result = await process_github_repository(
                                api,
                                url,
                                version,
                                name=name
                            )
                    # 新增：保留 input_name 字段
                    result["input_name"] = name
                    result["input_url"] = original_url  # 确保始终使用原始URL
                    results[index] = result
                except Exception as e:
                    logger.error(f"处理失败 {row.get('github_url')}: {e}", exc_info=True)
                    results[index] = {
                        "input_url": row.get("github_url"), 
                        "status": "error", 
                        "error": str(e),
                        "input_name": name  # 错误时也保留 input_name
                    }
                finally:
                    running_tasks -= 1
                    logger.info(f"任务完成，当前并发任务数: {running_tasks}")

                # 更新进度条
                pbar.update(1)

                # 检查是否需要保存临时文件
                current_time = time.time()
                if current_time - last_save_time >= SAVE_INTERVAL:
                    await save_temp_results()
                    last_save_time = current_time

        tasks = [sem_task(row, idx) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)

    logger.info(f"[ASYNC] 并发任务数上限: {max_concurrency}")
    
    try:
        with tqdm(total=len(df), desc="处理进度") as pbar:
            await process_with_progress(pbar)
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        # 发生错误时也保存临时结果
        await save_temp_results()
        raise
    finally:
        # 确保在任何情况下都保存最终结果
        await save_temp_results()

    # 返回所有结果（按原始顺序）
    ordered_results = [results[i] for i in range(len(df)) if i in results]
    return ordered_results

async def initialize_api():
    """异步初始化 API 客户端"""
    api = GitHubAPI()
    await api.initialize()
    return api

async def main_async():
    """异步主函数"""
    load_dotenv(".env")
    loggers = setup_logging()
    logger = loggers["main"]
    
    try:
        # 初始化 GitHub API
        logger.info("Step 1: Initializing GitHub API client")
        api = await initialize_api()
        logger.info("GitHub API client initialized successfully")
        
        # 读取输入文件
        logger.info("Step 2: Reading input Excel file")
        df = pd.read_excel("input.xlsx")
        logger.info(f"Read {len(df)} rows from input file")
        
        # 并发处理仓库
        results = await process_all_repos(api, df, max_concurrency=MAX_CONCURRENCY)
        
        # 处理结果输出
        output_df = pd.DataFrame(results)
        
        # 添加 concluded_license
        logger.info("生成 concluded_license...")
        output_df['concluded_license'] = output_df.apply(
            lambda row: get_concluded_license(
                row.get('license_type'),
                row.get('readme_license'),
                row.get('license_file_license')
            ),
            axis=1
        )
        logger.info("生成 thirdparty_dirs 列...")
        output_df = extract_thirdparty_dirs_column(output_df)

        # 根据 thirdparty_dirs 追加 " AND Others"
        logger.info("根据 thirdparty_dirs 更新 concluded_license...")
        def _has_thirdparty(row):
            # 优先检查 license_analysis.thirdparty_dirs
            analysis = row.get("license_analysis")
            if isinstance(analysis, dict):
                dirs = analysis.get("thirdparty_dirs")
                if isinstance(dirs, list) and len(dirs) > 0:
                    return True
            # 其次检查已生成的 thirdparty_dirs 列（非空字符串视为包含）
            tp_col = row.get("thirdparty_dirs")
            return isinstance(tp_col, str) and tp_col.strip() != ""

        def _append_others(expr: str) -> str:
            expr = (expr or "").strip()
            if not expr:
                return "Others"
            if not expr.endswith(" AND Others"):
                return f"{expr} AND Others"
            return expr

        output_df["concluded_license"] = output_df.apply(
            lambda r: _append_others(r.get("concluded_license"))
            if _has_thirdparty(r) else r.get("concluded_license"),
            axis=1
        )
        
        # 重排列顺序
        logger.info("重排列顺序...")
        # 获取实际存在的列（配置的列和实际数据的交集）
        existing_columns = [col for col in RESULT_COLUMNS_ORDER if col in output_df.columns]
        # 添加任何在数据中存在但不在配置中的列
        remaining_columns = [col for col in output_df.columns if col not in existing_columns]
        # 合并所有列
        final_columns = existing_columns + remaining_columns
        output_df = output_df[final_columns]
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"output_{timestamp}.xlsx")
        logger.info(f"保存结果到: {output_file}")
        
        # 修改保存逻辑，添加工作表名称
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)
            
        # 同时保存一个最新版本
        latest_file = os.path.join(output_dir, "output_latest.xlsx")
        with pd.ExcelWriter(latest_file, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)

        logger.info(f"处理完成! 共处理 {len(results)} 个仓库")
        logger.info(f"结果已保存到: {output_file}")
        logger.info(f"最新结果副本已保存到: {latest_file}")
        
    except Exception as e:
        logger.error(f"Error in main_async: {str(e)}", exc_info=True)
        raise

def main():
    """同步入口函数"""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logger.error(f"Failed to run main_async: {str(e)}", exc_info=True)
        sys.exit(1)

# ============================================================================
# Script Entry Point
# ============================================================================
# Standard Python script entry point that calls the main function

if __name__ == "__main__":
    main()


