import os
import re
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
from typing import List, Dict, Any, Optional

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

class PyPIAPIError(Exception):
    """PyPI API 调用异常"""
    pass

def _create_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """创建带重试机制的 Session"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504, 404],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # 设置代理（如果环境变量中有配置）
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    if http_proxy or https_proxy:
        session.proxies = {
            'http': http_proxy,
            'https': https_proxy or http_proxy
        }
    
    return session

def _fetch_pypi_metadata(package_name: str, max_retries: int = 3) -> Dict[str, Any]:
    """获取 PyPI 包的元数据，带重试机制"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    session = _create_session(retries=max_retries)
    timeout = (5, 15)  # (连接超时, 读取超时)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to fetch PyPI metadata for {package_name} (attempt {attempt + 1}/{max_retries})")
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL verification failed (attempt {attempt + 1}), retrying without verification")
            try:
                response = session.get(url, verify=False, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as ssl_e:
                if attempt == max_retries - 1:
                    raise PyPIAPIError(f"SSL Error: {str(ssl_e)}")
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise PyPIAPIError(f"Timeout fetching metadata for {package_name}")
            logger.warning(f"Timeout fetching metadata (attempt {attempt + 1}), retrying...")
            time.sleep(attempt * 2)  # 指数退避
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise PyPIAPIError(f"Failed to fetch PyPI metadata: {str(e)}")
            logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
            time.sleep(attempt * 2)

async def process_pypi_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    """处理 PyPI 仓库信息"""
    logger.info(f"Starting PyPI repository processing: {url}")
    
    try:
        # 1. 解析包名
        package_name = _parse_package_name(url)
        if not package_name:
            return {"status": "error", "error": "Invalid PyPI URL", "input_url": url}
        
        # 2. 获取元数据
        try:
            metadata = _fetch_pypi_metadata(package_name)
        except PyPIAPIError as e:
            logger.error(f"PyPI API error for {package_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "input_url": url,
                "component_name": package_name
            }
            
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
        
        # 在调用 GitHub API 时也添加重试逻辑
        if repo_url and "github.com" in repo_url:
            for attempt in range(3):  # GitHub API 重试3次
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
                    break  # 成功则退出重试
                except Exception as e:
                    logger.warning(f"GitHub API attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:  # 最后一次尝试失败
                        logger.error(f"Failed to process GitHub repository after 3 attempts: {str(e)}")
                    else:
                        time.sleep(attempt * 2)  # 重试等待
                        
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