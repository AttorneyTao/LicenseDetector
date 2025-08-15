import os
import re
import json
import logging
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .config import GEMINI_CONFIG
import google.generativeai as genai
from .utils import analyze_license_content, extract_copyright_info, analyze_license_content_async

# 日志设置
logger = logging.getLogger('main')
llm_logger = logging.getLogger('llm_interaction')

def _parse_package_name(url: str) -> str:
    """从 PyPI URL 中提取包名"""
    path = urlparse(url).path
    parts = [p for p in path.split('/') if p]
    if len(parts) >= 2 and parts[0] == "project":
        return parts[1]
    return parts[0] if parts else ""

def _fetch_pypi_metadata(package_name: str) -> Dict[str, Any]:
    """获取 PyPI 包的元数据"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

async def process_pypi_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    """处理 PyPI 仓库信息，返回格式与 process_npm_repository 一致"""
    logger.info(f"Starting PyPI repository processing: {url}")
    
    try:
        # 1. 解析包名
        package_name = _parse_package_name(url)
        if not package_name:
            return {"status": "error", "error": "Invalid PyPI URL"}
        
        # 2. 获取元数据
        metadata = _fetch_pypi_metadata(package_name)
        
        # 3. 版本处理
        if version and version in metadata["releases"]:
            resolved_version = version
        else:
            resolved_version = metadata["info"]["version"]  # 最新版本
        
        # 4. 获取版本特定信息
        version_info = next((r for r in metadata["releases"][resolved_version] 
                           if r["packagetype"] == "sdist"), 
                          metadata["releases"][resolved_version][0])
        
        # 5. 基本信息提取
        info = metadata["info"]
        license_type = info.get("license") or "Unknown"
        readme_content = info.get("description", "")
        
        # 6. 源码仓库 URL 处理
        repo_url = None
        if info.get("project_urls"):
            for key, value in info["project_urls"].items():
                if "github.com" in value.lower():
                    repo_url = value
                    break
        if not repo_url and "github.com" in (info.get("home_page") or ""):
            repo_url = info["home_page"]
            
        # 7. 调用 GitHub API 补充信息（如果有 GitHub 仓库）
        github_fields = {
            "license_files": None,
            "license_analysis": None,
            "has_license_conflict": None,
            "readme_license": None,
            "license_file_license": None
        }
        
        if repo_url and "github.com" in repo_url:
            try:
                from core.github_utils import process_github_repository, GitHubAPI
                api = GitHubAPI()
                github_result = await process_github_repository(
                    api,
                    repo_url,
                    resolved_version
                )
                if github_result and github_result.get("status") != "error":
                    for key in ["license_analysis", "has_license_conflict", 
                              "readme_license", "license_file_license"]:
                        if github_result.get(key) is not None:
                            github_fields[key] = github_result[key]
            except Exception as e:
                logger.error(f"Failed to process GitHub repository: {str(e)}")
        
        # 8. 处理版权信息
        author = info.get("author", "")
        if not author:
            author = f"{package_name} original author and authors"
        
        copyright_notice = extract_copyright_info(readme_content)
        if not copyright_notice:
            current_year = datetime.now(timezone.utc).year
            copyright_notice = f"Copyright (c) {current_year} {author}"
        
        # 9. 返回结果
        result = {
            "input_url": url,
            "repo_url": repo_url,
            "input_version": version,
            "resolved_version": resolved_version,
            "used_default_branch": version is None,
            "component_name": package_name,
            "license_files": f"https://pypi.org/project/{package_name}/{resolved_version}/#files",
            "license_analysis": github_fields["license_analysis"],
            "license_type": license_type,
            "has_license_conflict": github_fields["has_license_conflict"],
            "readme_license": github_fields["readme_license"],
            "license_file_license": github_fields["license_file_license"],
            "copyright_notice": copyright_notice,
            "status": "success",
            "license_determination_reason": "Fetched from PyPI registry",
            "readme": readme_content[:5000] if readme_content else None
        }
        
        logger.info(f"Processing completed for PyPI package: {package_name}@{resolved_version}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PyPI repository: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "input_url": url
        }