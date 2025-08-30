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
import google.generativeai as genai

#=============================================================================
# Import internal packages
#=============================================================================
from core.logging_utils import setup_logging
from core.github_utils import GitHubAPI
from core.config import GEMINI_CONFIG, SCORE_THRESHOLD, MAX_CONCURRENCY, RESULT_COLUMNS_ORDER
from core.utils import get_concluded_license

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



# Validate Gemini API configuration only if LLM is enabled
if USE_LLM:
    if not GEMINI_CONFIG["api_key"]:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Initialize Gemini client
    genai.configure(api_key=GEMINI_CONFIG["api_key"])
    logger.info(f"Initialized Gemini API with model: {GEMINI_CONFIG['model']}")





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
                try:
                    running_tasks += 1
                    logger.info(f"当前并发任务数: {running_tasks}")
                    
                    result = await process_github_repository(
                        api,
                        row["github_url"],
                        row.get("version"),
                        name=row.get("name", None)
                    )
                    results[index] = result
                except Exception as e:
                    logger.error(f"处理失败 {row.get('github_url')}: {e}", exc_info=True)
                    results[index] = {
                        "input_url": row.get("github_url"), 
                        "status": "error", 
                        "error": str(e)
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


