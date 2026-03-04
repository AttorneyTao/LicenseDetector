"""
FastAPI application for exposing license analysis as a service
"""

import os
import sys
import logging
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import existing modules
from core.github_utils import GitHubAPI, process_github_repository
from core.config import MAX_CONCURRENCY, RESULT_COLUMNS_ORDER
from core.utils import get_concluded_license, extract_thirdparty_dirs_column
from core.logging_utils import setup_logging
from core.email_utils import send_analysis_result, EmailConfig
from core.go_utils import get_github_url_from_pkggo
from core.maven_utils import analyze_maven_repository_url

import pandas as pd
from tqdm import tqdm

# Setup logging
loggers = setup_logging()
logger = loggers.get("main", logging.getLogger(__name__))

# Create FastAPI app
app = FastAPI(
    title="GitHub License Analyzer API",
    description="API for analyzing GitHub repository licenses",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance
_api_instance = None


async def get_github_api():
    """Get or create GitHub API instance"""
    global _api_instance
    if _api_instance is None:
        _api_instance = GitHubAPI()
        await _api_instance.initialize()
    return _api_instance


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("API启动中...")
    try:
        api = await get_github_api()
        logger.info("GitHub API客户端初始化成功")
    except Exception as e:
        logger.error(f"API初始化失败: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API关闭中...")
    global _api_instance
    if _api_instance:
        await _api_instance.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "GitHub License Analyzer API"
    }


@app.post("/api/v1/analyze")
async def analyze_licenses(
    file: UploadFile = File(...),
    email: str = Form(...),
    smtp_server: Optional[str] = Form(None),
    smtp_port: Optional[int] = Form(None),
):
    """
    分析许可证信息并发送结果到邮箱
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        email: 接收结果的邮箱地址
        smtp_server: SMTP服务器地址（可选，使用环境变量）
        smtp_port: SMTP端口（可选，使用环境变量）
        
    Returns:
        分析结果和邮件发送状态
    """
    temp_input = None
    temp_output = None
    
    try:
        # 验证邮箱格式
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="无效的邮箱地址")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
            temp_input = tmp_input.name
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()
        
        logger.info(f"接收到文件: {file.filename}, 保存至: {temp_input}")
        
        # 读取Excel文件
        try:
            df = pd.read_excel(temp_input)
            logger.info(f"读取了 {len(df)} 行数据")
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"无法读取Excel文件: {str(e)}"
            )
        
        # 初始化API
        api = await get_github_api()
        
        # 处理仓库
        logger.info("开始处理仓库...")
        results = await _process_repositories(api, df)
        
        # 生成输出
        logger.info("生成输出文件...")
        output_df = _generate_output(results)
        
        # 保存输出文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
            temp_output = tmp_output.name
        
        with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)
        
        logger.info(f"输出文件已保存: {temp_output}")
        
        # 发送邮件
        logger.info(f"发送结果到邮箱: {email}")
        email_config = None
        if smtp_server or smtp_port:
            email_config = EmailConfig(
                smtp_server=smtp_server,
                smtp_port=smtp_port
            )
        
        email_sent = send_analysis_result(
            recipient_email=email,
            output_file_path=temp_output,
            smtp_config=email_config
        )
        
        # 返回结果
        return {
            "status": "success",
            "message": "分析完成" + ("，结果已发送至邮箱" if email_sent else "，但邮件发送失败"),
            "processed_rows": len(results),
            "email_sent": email_sent,
            "email": email,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"已删除临时输入文件: {temp_input}")
            except Exception as e:
                logger.warning(f"删除临时输入文件失败: {e}")
        
        if temp_output and os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
                logger.debug(f"已删除临时输出文件: {temp_output}")
            except Exception as e:
                logger.warning(f"删除临时输出文件失败: {e}")


@app.post("/api/v1/analyze-and-download")
async def analyze_and_download(
    file: UploadFile = File(...)
):
    """
    分析许可证信息并返回Excel文件（不发送邮件）
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        
    Returns:
        输出Excel文件
    """
    temp_input = None
    temp_output = None
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
            temp_input = tmp_input.name
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()
        
        logger.info(f"接收到文件: {file.filename}, 保存至: {temp_input}")
        
        # 读取Excel文件
        try:
            df = pd.read_excel(temp_input)
            logger.info(f"读取了 {len(df)} 行数据")
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"无法读取Excel文件: {str(e)}"
            )
        
        # 初始化API
        api = await get_github_api()
        
        # 处理仓库
        logger.info("开始处理仓库...")
        results = await _process_repositories(api, df)
        
        # 生成输出
        logger.info("生成输出文件...")
        output_df = _generate_output(results)
        
        # 保存输出文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
            temp_output = tmp_output.name
        
        with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)
        
        logger.info(f"输出文件已保存: {temp_output}")
        
        # 返回文件
        return FileResponse(
            path=temp_output,
            filename=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        # 注意：FileResponse会自动清理文件，所以这里只清理输入文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"已删除临时输入文件: {temp_input}")
            except Exception as e:
                logger.warning(f"删除临时输入文件失败: {e}")


async def _process_repositories(api, df):
    """Process all repositories with concurrency control"""
    from core.github_utils import normalize_github_url
    import re
    
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = {}
    
    async def process_single(row, index):
        async with sem:
            name = None
            try:
                original_url = row.get("github_url")
                url = normalize_github_url(original_url)
                version = row.get("version")
                name = row.get("name", None)
                
                # Check if it's a Go package
                is_go_pkg = False
                if isinstance(url, str):
                    if url.startswith("https://pkg.go.dev/") or url.startswith("https://go.dev/") or re.match(r"^go\.[\w\.-]+/", url):
                        is_go_pkg = True
                
                if is_go_pkg:
                    logger.info(f"检测到 Go 包 URL: {url}")
                    github_info = await get_github_url_from_pkggo(url, version, name)
                    github_url = github_info.get("github_url")
                    if github_url:
                        result = await process_github_repository(api, github_url, version, name=name)
                    else:
                        result = await process_github_repository(api, url, version, name=name)
                else:
                    # Check if it's a Maven URL
                    is_maven_url = isinstance(url, str) and (
                        "mvnrepository.com/artifact" in url or 
                        "repo1.maven.org/maven2" in url
                    )
                    if is_maven_url:
                        logger.info(f"检测到 Maven URL: {url}")
                        result = await process_github_repository(api, url, version, name=name)
                        
                        if result.get("status") != "success":
                            logger.info(f"GitHub 流程未成功，调用 Maven 处理函数")
                            try:
                                maven_result = analyze_maven_repository_url(url)
                                license_file_url = f"https://mvnrepository.com/artifact/{maven_result['group_id']}/{maven_result['artifact_id']}/{maven_result.get('version', '')}"
                                
                                copyright_notice = maven_result.get('copyright')
                                if not copyright_notice:
                                    org_parts = maven_result['group_id'].split(".")
                                    orgname = org_parts[1] if len(org_parts) > 1 else org_parts[0]
                                    copyright_notice = f"Copyright (c) {datetime.now().year} {orgname.capitalize()}"
                                
                                result = {
                                    "input_url": original_url,
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
                                    "copyright_notice": copyright_notice,
                                    "status": "success",
                                    "input_name": name,
                                }
                            except Exception as e:
                                logger.warning(f"Maven 处理失败: {e}")
                    else:
                        result = await process_github_repository(api, url, version, name=name)
                
                result["input_name"] = name
                result["input_url"] = original_url
                results[index] = result
                
            except Exception as e:
                logger.error(f"处理失败 {row.get('github_url')}: {e}", exc_info=True)
                results[index] = {
                    "input_url": row.get("github_url"),
                    "status": "error",
                    "error": str(e),
                    "input_name": name
                }
    
    # Process all rows
    tasks = [process_single(row, idx) for idx, row in df.iterrows()]
    await asyncio.gather(*tasks)
    
    # Return ordered results
    return [results[i] for i in range(len(df)) if i in results]


def _generate_output(results):
    """Generate output DataFrame from results"""
    output_df = pd.DataFrame(results)
    
    # Add concluded_license
    output_df['concluded_license'] = output_df.apply(
        lambda row: get_concluded_license(
            row.get('license_type'),
            row.get('readme_license'),
            row.get('license_file_license')
        ),
        axis=1
    )
    
    # Extract thirdparty_dirs
    output_df = extract_thirdparty_dirs_column(output_df)
    
    # Append " AND Others" if thirdparty_dirs exist
    def _has_thirdparty(row):
        analysis = row.get("license_analysis")
        if isinstance(analysis, dict):
            dirs = analysis.get("thirdparty_dirs")
            if isinstance(dirs, list) and len(dirs) > 0:
                return True
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
    
    # Reorder columns
    existing_columns = [col for col in RESULT_COLUMNS_ORDER if col in output_df.columns]
    remaining_columns = [col for col in output_df.columns if col not in existing_columns]
    final_columns = existing_columns + remaining_columns
    output_df = output_df[final_columns]
    
    return output_df


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn
    
    logger.info(f"启动API服务器: {host}:{port}")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run_api_server()
