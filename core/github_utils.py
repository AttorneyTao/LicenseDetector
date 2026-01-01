
from datetime import datetime
import os
import logging
from urllib.parse import urlparse
import requests
import time
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import yaml
from dotenv import load_dotenv
import re
import json
import httpx  # 新增：导入 httpx
import asyncio  # 新增：导入 asyncio
from httpx import AsyncClient
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from core.npm_utils import process_npm_repository
from core.pypi_utils import process_pypi_repository
from core.utils import analyze_license_content, construct_copyright_notice, find_license_files, find_readme, find_top_level_thirdparty_dirs, is_sha_version, analyze_license_content_async, construct_copyright_notice_async, find_license_files_detailed
from core.nuget_utils import process_nuget_packages, check_if_nuget_package_exists
from core.llm_provider import get_llm_provider
import platform
from openai import AsyncOpenAI

class Kind(Enum):
    REPO = "REPO"
    DIR = "DIR"
    FILE = "FILE"


#==============================================================================
from .config import LLM_CONFIG

load_dotenv()
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

logger = logging.getLogger('main')
llm_logger = logging.getLogger('llm_interaction')

def normalize_github_url(url: str) -> str:
    """如果是 github.com/xxx/yyy 但没有协议头，自动补全为 https://github.com/xxx/yyy"""
    if not isinstance(url, str):
        return "" if url is None else str(url)
    url = url.strip()
    if url.startswith("github.com/"):
        return "https://" + url
    return url

class GitHubAPI:
    """
    A client class for interacting with the GitHub API.
    
    This class provides methods to interact with GitHub's REST API, including:
    - Authentication using GitHub tokens with proper header management
    - Rate limit handling with automatic retry and wait mechanisms
    - Repository information retrieval including metadata and structure
    - License information retrieval with fallback mechanisms
    - Branch and tag management with version resolution
    - Proxy support with automatic fallback to direct connection
    
    The class implements robust error handling and logging for all operations.
    
    Attributes:
        BASE_URL (str): The base URL for GitHub's API (https://api.github.com)
        token (str): GitHub authentication token loaded from environment
        headers (dict): HTTP headers including authentication and API version
        session (requests.Session): HTTP session for making requests with proxy support
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self):  # 改回同步初始化
        """同步初始化基本属性"""
        import requests
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {os.getenv('GITHUB_TOKEN')}"
        }
        # 初始化 session
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    async def initialize(self):  # 新增异步初始化方法
        """异步初始化和测试连接"""
        try:
            logger.info("Testing GitHub API connection...")
            await self._make_request("/rate_limit")
            logger.info("GitHub API connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to GitHub API: {str(e)}")
            raise

    def _make_request_sync(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Makes an authenticated request to the GitHub API with rate limit handling.
        
        This method implements:
        - Automatic rate limit detection and handling
        - Request retry logic with exponential backoff
        - Proper error handling and logging
        - Response validation and JSON parsing
        
        Args:
            endpoint (str): The API endpoint to request (e.g., '/repos/owner/repo')
            params (Optional[Dict]): Query parameters for the request
                Example: {'ref': 'main', 'recursive': '1'}
            
        Returns:
            Dict: The JSON response from the API
            
        Raises:
            requests.exceptions.HTTPError: If the request fails with a non-200 status code
            requests.exceptions.RequestException: For network or connection errors
            ValueError: If the response cannot be parsed as JSON
        """
        # Construct the full API URL
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Making request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        # Implement rate limit handling with retry logic
        while True:
            response = self.session.get(url, params=params)
            # Check for rate limit exceeded
            if response.status_code == 403 and "rate limit" in response.text.lower():
                # Calculate wait time based on rate limit reset time
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            logger.debug(f"Request successful. Status code: {response.status_code}")
            return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        异步请求处理，包含速率限制处理和自动重定向
        """
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Making request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        async with AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url, 
                params=params,
                headers=self.headers
            )
            # 如果不是 2xx，记录并抛出异常
            if response.status_code in (301, 302, 307, 308):
                logger.warning(f"Redirected ({response.status_code}) to: {response.headers.get('location')}")
            if not (200 <= response.status_code < 300):
                logger.error(f"GitHub API request failed: {response.status_code} {response.text}")
                response.raise_for_status()
            return response.json()
    async def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """获取文件内容，包含超时和重试机制"""
        max_retries = 3
        timeout = 30.0
        
        for attempt in range(max_retries):
            try:
                # 获取代理设置
                proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
                
                # 配置超时和代理设置
                client_kwargs = {
                    'timeout': timeout,
                    'limits': httpx.Limits(max_keepalive_connections=5, max_connections=10)
                }
                
                # 如果有代理设置，添加到client配置中
                if proxy:
                    client_kwargs['proxies'] = proxy
                
                async with AsyncClient(**client_kwargs) as client:
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
                    logger.debug(f"Attempting to fetch: {raw_url} (attempt {attempt + 1}/{max_retries})")
                    
                    # 发起请求（不在get方法中传递proxies参数）
                    response = await client.get(raw_url)
                    response.raise_for_status()
                    return response.text
                    
            except httpx.ConnectTimeout as e:
                logger.warning(f"Connection timeout on attempt {attempt + 1}/{max_retries} for {path}: {str(e)}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    logger.error(f"All {max_retries} attempts failed for {path}")
                    return None
                # 等待后重试
                await asyncio.sleep(2 ** attempt)  # 指数退避
                
            except httpx.RequestError as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries} for {path}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Unexpected error fetching {path}: {str(e)}")
                return None

        return None

    def get_repo_info_sync(self, owner: str, repo: str) -> Dict:
        """
        Retrieves detailed information about a GitHub repository.
        
        This method fetches:
        - Repository metadata (name, description, etc.)
        - Default branch information
        - Repository statistics
        - Owner information
        - Repository settings
        
        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name
            
        Returns:
            Dict: Repository information including:
                - name: Repository name
                - description: Repository description
                - default_branch: Default branch name
                - owner: Owner information
                - created_at: Creation timestamp
                - updated_at: Last update timestamp
                - stars_count: Number of stars
                - forks_count: Number of forks
        """
        logger.info(f"Fetching repository info for {owner}/{repo}")
        result = self._make_request(f"/repos/{owner}/{repo}")
        return result if isinstance(result, dict) else {}

    async def get_repo_info(self, owner: str, repo: str) -> Dict:
        """
        异步获取仓库信息
        """
        logger.info(f"Fetching repository info for {owner}/{repo}")
        try:
            return await self._make_request(f"/repos/{owner}/{repo}")
        except Exception as e:
            logger.error(f"Error fetching repo info for {owner}/{repo}: {str(e)}")
            raise

    def get_branches_sync(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Retrieves all branches for a repository.
        
        This method fetches:
        - Branch names
        - Commit SHAs
        - Protection rules
        - Latest commit information
        
        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name
            
        Returns:
            List[Dict]: List of branch information, each containing:
                - name: Branch name
                - commit: Latest commit information
                - protected: Whether branch is protected
                - protection_url: URL to protection rules
        """
        logger.info(f"Fetching branches for {owner}/{repo}")
        result = self._make_request(f"/repos/{owner}/{repo}/branches")
        return result if isinstance(result, list) else []

    async def get_branches(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching branches for {owner}/{repo}")
        result = await self._make_request(f"/repos/{owner}/{repo}/branches")
        return result if isinstance(result, list) else []

    def get_tags_sync(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tags for a repository, handling pagination.

        This method fetches:
        - Tag names
        - Commit SHAs
        - Tag creation information
        - Tag message/description

        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name

        Returns:
            List[Dict]: List of tag information, each containing:
                - name: Tag name
                - commit: Commit information
                - zipball_url: URL to download zip archive
                - tarball_url: URL to download tar archive
        """
        logger.info(f"Fetching tags for {owner}/{repo} (with pagination)")
        tags: List[Dict] = []
        page = 1
        per_page = 100
        
        while True:
            response = self._make_request(
                f"/repos/{owner}/{repo}/tags",
                params={"per_page": per_page, "page": page}
            )
            if not response:
                break
            if isinstance(response, dict):
                # Defensive: sometimes API returns dict with 'message' on error
                logger.warning(f"Unexpected response format when fetching tags: {response}")
                break
            if isinstance(response, list):
                tags.extend(response)
            if isinstance(response, list) and len(response) < per_page:
                break
            page += 1
        return tags

    async def get_tags(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        异步获取所有 tags
        """
        logger.info(f"Fetching tags for {owner}/{repo} (with pagination)")
        tags = []
        page = 1
        per_page = 100
        
        while True:
            response = await self._make_request(
                f"/repos/{owner}/{repo}/tags",
                params={"per_page": per_page, "page": page}
            )
            if not response:
                break
            if isinstance(response, dict):
                logger.warning(f"Unexpected response format when fetching tags: {response}")
                break
            if isinstance(response, list):
                tags.extend(response)
            if isinstance(response, list) and len(response) < per_page:
                break
            page += 1
        return tags

    def get_tree_sync(self, owner: str, repo: str, sha: str) -> Dict:
        """
        Retrieves the complete repository tree structure.
        
        This method fetches:
        - File and directory structure
        - File modes and types
        - File sizes
        - SHA hashes
        - URLs for each item
        
        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name
            sha (str): Commit SHA or branch name to get tree for
            
        Returns:
            Dict: Repository tree structure containing:
                - sha: Tree SHA
                - url: API URL for the tree
                - tree: List of tree items, each containing:
                    - path: File/directory path
                    - mode: File mode
                    - type: Item type (blob/tree)
                    - sha: Item SHA
                    - size: File size (for blobs)
                    - url: API URL for the item
        """
        logger.info(f"Fetching tree for {owner}/{repo} at {sha}")
        result = self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})
        return result if isinstance(result, dict) else {}

    async def get_tree(self, owner: str, repo: str, sha: str) -> Dict:
        logger.info(f"Fetching tree for {owner}/{repo} at {sha}")
        return await self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})

    def get_license_sync(self, owner: str, repo: str, ref: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieves license information for a repository.
        
        This method:
        - Checks for LICENSE file in repository
        - Detects license type using GitHub's license detection
        - Retrieves license content
        - Provides license metadata
        
        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name
            ref (Optional[str]): Reference (branch/tag/commit) to check license for
            
        Returns:
            Optional[Dict]: License information if found, containing:
                - name: License name
                - path: Path to license file
                - sha: License file SHA
                - size: License file size
                - url: URL to license file
                - html_url: Web URL for license file
                - git_url: Git URL for license file
                - download_url: Raw content URL
                - type: File type
                - content: Base64 encoded content
                - encoding: Content encoding
                - _links: Related URLs
                - license: License metadata including:
                    - key: License identifier
                    - name: License name
                    - spdx_id: SPDX identifier
                    - url: License URL
                    - node_id: GitHub node ID
            None if no license is found
        """
        logger.info(f"Fetching license info for {owner}/{repo} at ref: {ref}")
        try:
            # Add ref parameter if specified
            params = {"ref": ref} if ref else None
            result = self._make_request(f"/repos/{owner}/{repo}/license", params=params)
            return result if isinstance(result, dict) else None
        except Exception as e:
            # Handle 404 (no license found) gracefully
            if "404" in str(e):
                logger.warning(f"No license found for {owner}/{repo}")
                return None
            logger.error(f"Error fetching license: {str(e)}")
            raise
    async def get_license(self, owner: str, repo: str, ref: Optional[str] = None) -> Optional[Dict]:
        logger.info(f"Fetching license info for {owner}/{repo} at ref: {ref}")
        try:
            params = {"ref": ref} if ref else None
            return await self._make_request(f"/repos/{owner}/{repo}/license", params=params)
        except Exception as e:
            # 检查是否是HTTP 404错误
            if "404" in str(e):
                logger.warning(f"No license found for {owner}/{repo}")
                return None
            logger.error(f"Error fetching license: {str(e)}")
            raise

def find_github_url_from_package_url_sync(package_url: str) -> Optional[str]:
    """
    Attempts to find a GitHub URL from a package URL.
    
    This function:
    - Uses LLM to analyze package URL
    - Supports multiple package managers
    - Provides confidence scores
    - Handles different URL formats
    
    Args:
        package_url (str): Package URL
            Examples:
            - npm: https://www.npmjs.com/package/package-name
            - maven: https://mvnrepository.com/artifact/group/artifact
            - pypi: https://pypi.org/project/package-name
            
    Returns:
        Optional[str]: GitHub URL if found, None otherwise
            Example: "https://github.com/owner/repo"
    """
    if not USE_LLM:
        logger.info("LLM is disabled, skipping GitHub URL lookup")
        return None
        
    try:
        # prompt = """
        # Given the following package URL, find the corresponding GitHub repository URL if it exists.
        # Package URL: {package_url}
        
        # Return the result in JSON format:
        # {{
        #     "github_url": "https://github.com/owner/repo if found, otherwise null",
        #     "confidence": 0.0-1.0
        # }}
        
        # Only return a GitHub URL if you are confident it is the correct repository.
        # If you are not sure, return null.
        # """
        prompt = PROMPTS["github_url_finder"].format(package_url=package_url)
        
        llm_logger.info("GitHub URL Lookup Request:")
        llm_logger.info(f"Prompt: {prompt}")
        
        provider = get_llm_provider()
        response = provider.generate(prompt)
        
        llm_logger.info("GitHub URL Lookup Response:")
        llm_logger.info(f"Response: {response}")
        
        if response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                github_url = result.get("github_url")
                confidence = result.get("confidence", 0.0)
                
                llm_logger.info(f"Found GitHub URL: {github_url} with confidence {confidence}")
                
                if github_url and confidence >= 0.7:  # Only accept if confidence is high enough
                    return github_url
                else:
                    llm_logger.info(f"No confident GitHub URL match found (confidence: {confidence})")
            else:
                llm_logger.warning("No JSON found in GitHub URL lookup response")
    except Exception as e:
        llm_logger.error(f"Failed to find GitHub URL: {str(e)}", exc_info=True)
    return None


async def select_primary_license_file(license_files_detailed: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    使用LLM选择最可能的主要项目级license文件

    Args:
        license_files_detailed: license文件详细信息列表，每个元素包含path, filename, url, directory

    Returns:
        选中的主要license文件信息，如果没有或选择失败则返回None
    """
    if not license_files_detailed:
        return None
    
    if len(license_files_detailed) == 1:
        return license_files_detailed[0]
    
    if not USE_LLM:
        logger.info("LLM is disabled, using first license file as fallback")
        return license_files_detailed[0]
    
    # 准备license文件信息给LLM
    license_info_text = []
    for i, file_info in enumerate(license_files_detailed, 1):
        path = file_info['path']
        directory = file_info['directory']
        filename = file_info['filename']
        
        info_line = f"{i}. Path: {path}"
        if directory:
            info_line += f" (Directory: {directory})"
        else:
            info_line += " (Root directory)"
        info_line += f" | Filename: {filename}"
        license_info_text.append(info_line)
    
    license_files_info = "\n".join(license_info_text)
    
    prompt = PROMPTS["license_priority_selector"].format(license_files_info=license_files_info)
    
    llm_logger.info("License Priority Selection Request:")
    llm_logger.info(f"Prompt: {prompt}")
    
    try:
        provider = get_llm_provider()
        response = provider.generate(prompt)
        
        llm_logger.info("License Priority Selection Response:")
        llm_logger.info(f"Response: {response}")
        
        if response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                primary_license_path = result.get("primary_license_path")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                llm_logger.info(f"Selected primary license: {primary_license_path} with confidence {confidence}")
                llm_logger.info(f"Reasoning: {reasoning}")
                
                # 查找匹配的license文件
                for file_info in license_files_detailed:
                    if file_info['path'] == primary_license_path:
                        logger.info(f"Successfully selected primary license file: {primary_license_path}")
                        return file_info
                
                logger.warning(f"LLM selected path {primary_license_path} not found in license files list")
            else:
                llm_logger.warning("No JSON found in license priority selection response")
    except Exception as e:
        # 特别处理异步事件循环错误
        if "This event loop is already running" in str(e):
            llm_logger.warning("Event loop conflict detected - falling back to heuristic selection")
        else:
            llm_logger.error(f"Failed to select primary license file: {str(e)}", exc_info=True)
    
    # 降级处理：如果LLM选择失败，使用简单的启发式规则
    logger.info("Falling back to heuristic license file selection")
    
    # 优先选择根目录的license文件
    root_licenses = [f for f in license_files_detailed if not f['directory']]
    if root_licenses:
        # 在根目录license中，优先选择标准命名的文件
        priority_names = ['license', 'license.txt', 'license.md', 'copying', 'copying.txt']
        for priority_name in priority_names:
            for file_info in root_licenses:
                if file_info['filename'].lower() == priority_name:
                    logger.info(f"Selected license file by heuristic: {file_info['path']}")
                    return file_info
        # 如果没有标准命名，返回第一个根目录license
        logger.info(f"Selected first root license file: {root_licenses[0]['path']}")
        return root_licenses[0]
    
    # 如果没有根目录license，返回第一个
    logger.info(f"Selected first available license file: {license_files_detailed[0]['path']}")
    return license_files_detailed[0]


async def resolve_github_version(api: GitHubAPI, owner: str, repo: str, version: Optional[str]) -> Tuple[str, bool]:
    """
    First try text matching, fallback to Gemini LLM if no match.
    Supports "0.x" style ranges, "v" prefix, and case-insensitive matching.
    """
    version_resolve_logger = logging.getLogger('version_resolve')
    version_resolve_logger.info(f"Resolving version for {owner}/{repo}, requested version: {version}")

    # 新增：如果version是SHA，直接返回
    if version and is_sha_version(version):
        version_resolve_logger.info(f"Version {version} detected as SHA, using directly.")
        return version, False

    # 新增：处理 v0.0.0-20200907205600-7a23bdc65eef 或 0.0.0-20200907205600-7a23bdc65eef 格式
    if version:
        version_str = str(version).strip()
        # 检查是否匹配模式：可选的'v'前缀 + 0.0.0-时间戳-SHA格式
        match = re.match(r'^v?\d+\.\d+\.\d+-\d{14}-([a-f0-9]+)$', version_str, re.IGNORECASE)
        if match:
            sha = match.group(1)[:7]  # 取第三节的前7位作为SHA
            version_resolve_logger.info(f"Version {version} detected as Go pseudo-version, extracted SHA: {sha}")
            return sha, False

    # Get default branch
    repo_info = await api.get_repo_info(owner, repo)  # 添加 await
    # 安全获取 default_branch，没有则 fallback
    default_branch: str = repo_info.get("default_branch", "main")
    if not default_branch:
        # fallback 顺序：main > master > 第一个 branch > 'main'
        branches = await api.get_branches(owner, repo)
        branch_names = [b.get("name") for b in branches if "name" in b]
        if "main" in branch_names:
            default_branch = "main"
        elif "master" in branch_names:
            default_branch = "master"
        elif branch_names:
            default_branch = branch_names[0] if branch_names[0] else "main"
        else:
            default_branch = "main"
        logging.getLogger('version_resolve').warning(
            f"repo_info for {owner}/{repo} missing 'default_branch', fallback to: {default_branch}"
        )
    version_resolve_logger.info(f"Repository default branch: {default_branch}")

    # Get all branches and tags
    branches = await api.get_branches(owner, repo)
    tags = await api.get_tags(owner, repo)
    def extract_name(item):
        if isinstance(item, dict) and "name" in item:
            return item["name"]
        elif isinstance(item, str):
            return item
        return None
    branch_names = [extract_name(b) for b in branches]
    tag_names = [extract_name(t) for t in tags]
    candidate_versions = [n for n in branch_names if n] + [n for n in tag_names if n]
    version_resolve_logger.info(f"Candidate versions: {candidate_versions}")

    # No version specified, use default branch
    if not version:
        version_resolve_logger.info("No version specified, using default branch")
        return default_branch, True

    version_str = str(version).strip()
    version_str_lower = version_str.lower().lstrip("v")

    # 1. Exact match ignoring "v" prefix and case
    for candidate in candidate_versions:
        cand_lower = candidate.lower().lstrip("v")
        if version_str_lower == cand_lower:
            version_resolve_logger.info(f"Found exact version match (ignore v/case): {candidate}")
            return candidate, False

    # 2. Range match like "0.x"
    if version_str_lower.endswith(".x"):
        base = version_str_lower[:-2]
        for candidate in candidate_versions:
            cand_lower = candidate.lower().lstrip("v")
            if cand_lower.startswith(base + "."):
                version_resolve_logger.info(f"Found version range match: {candidate} for {version_str}")
                return candidate, False

    # 3. Partial match (e.g. "1.2" matches "1.2.3")
    for candidate in candidate_versions:
        cand_lower = candidate.lower().lstrip("v")
        if version_str_lower in cand_lower:
            version_resolve_logger.info(f"Found partial version match: {candidate}")
            return candidate, False

    # 4. Fallback: Gemini LLM
    if USE_LLM:
        try:
            # prompt = f"""
            #             You are a GitHub repository version resolver.
            #             Here is the list of all available branches and tags (choose only from these):
            #             {candidate_versions}

            #             The user requested version string: {version}

            #             Please determine the most appropriate branch or tag name the user wants. Only return one value, and it must be strictly from the above list. Do not return SHA, explanations, or anything else.
            #             If you cannot determine or there is no suitable match, return "{default_branch}".

            #             Return in the following JSON format:
            #             {{
            #                 "resolved_version": "xxx",  // must be one of the candidates above
            #                 "used_default_branch": true/false  // whether the default branch was used
            #             }}
            # """
            prompt = PROMPTS["version_resolve"].format(
                candidate_versions=candidate_versions,
                version=version_str,
                default_branch=default_branch
            )

            llm_logger.info("Version Resolve Request:")
            llm_logger.info(f"Prompt: {prompt}")
            version_resolve_logger.info("Version Resolve LLM Request:")

            provider = get_llm_provider()
            response = provider.generate(prompt)
            llm_logger.info("Version Resolve Response:")
            llm_logger.info(f"Response: {response}")
            version_resolve_logger.info("Version Resolve LLM Response:")
            version_resolve_logger.info(f"Response: {response}")

            if response:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    resolved_version = result.get("resolved_version", default_branch)
                    used_default_branch = result.get("used_default_branch", resolved_version == default_branch)
                    logger.info(f"LLM resolved version: {resolved_version}, used_default_branch: {used_default_branch}")
                    version_resolve_logger.info(f"LLM resolved version: {resolved_version}, used_default_branch: {used_default_branch}")
                    return resolved_version, used_default_branch
                else:
                    version_resolve_logger.warning("No JSON found in version resolve response")
        except Exception as e:
            if "This event loop is already running" in str(e):
                version_resolve_logger.warning("Event loop conflict detected during version resolution - falling back to default branch")
            else:
                version_resolve_logger.error(f"Failed to resolve version via LLM: {str(e)}", exc_info=True)

    # fallback
    version_resolve_logger.info("No version matched, using default branch")
    return default_branch, True


def parse_github_url(url: str) -> Tuple[str, str, Kind]:
    url = normalize_github_url(url)
    """
    Parses a GitHub URL into its components.

    This function handles:
    - Standard GitHub repository URLs
    - URLs with specific branches/tags
    - URLs pointing to specific files
    - URLs pointing to directories
    - URLs with query parameters

    Args:
        url (str): GitHub URL to parse
            Examples:
            - https://github.com/owner/repo
            - https://github.com/owner/repo/tree/branch
            - https://github.com/owner/repo/blob/branch/file.txt

    Returns:
        Tuple[str, str, Kind]: Tuple containing:
            - Repository URL (e.g., https://github.com/owner/repo)
            - Subpath within repository (e.g., src/main.py)
            - Kind of URL (REPO/DIR/FILE)

    Raises:
        ValueError: If URL is not a valid GitHub URL or has invalid format
    """
    logger.info(f"Parsing GitHub URL: {url}")
    # 新增：去掉.git后缀
    if url.endswith(".git"):
        url = url[:-4]
        logger.debug(f"Removed .git suffix, new url: {url}")
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        logger.error(f"Invalid domain: {parsed.netloc}")
        raise ValueError("Not a GitHub URL")

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        logger.error(f"Invalid path parts: {path_parts}")
        raise ValueError("Invalid GitHub URL format")

    owner, repo = path_parts[:2]
    repo_url = f"https://github.com/{owner}/{repo}"

    if len(path_parts) <= 2:
        logger.debug(f"URL is a repository: {repo_url}")
        return repo_url, "", Kind.REPO

    sub_path = "/".join(path_parts[2:])
    if "." in path_parts[-1]:
        logger.debug(f"URL is a file: {repo_url}/{sub_path}")
        return repo_url, sub_path, Kind.FILE

    logger.debug(f"URL is a directory: {repo_url}/{sub_path}")
    return repo_url, sub_path, Kind.DIR


def draw_github_file_tree(tree_items: List[Dict], indent: str = "", is_last: bool = True, prefix: str = "") -> List[str]:
    """
    Generates a text representation of the repository file tree.

    This function:
    - Creates a hierarchical tree structure
    - Uses ASCII characters for tree visualization
    - Sorts items (directories first, then files)
    - Handles nested directories
    - Maintains proper indentation

    Args:
        tree_items (List[Dict]): List of tree items
            Each item should have:
            - path: Full path of the item
            - type: 'blob' for files, 'tree' for directories
        indent (str): Current indentation level
        is_last (bool): Whether this is the last item at current level
        prefix (str): Prefix for the current level

    Returns:
        List[str]: List of lines representing the tree structure
            Example:
            ├── src/
            │   ├── main.py
            │   └── utils.py
            └── README.md
    """
    lines = []

    # Sort items: directories first, then files, both alphabetically
    sorted_items = sorted(tree_items, key=lambda x: (x.get("type") != "tree", x.get("path", "").lower()))

    for i, item in enumerate(sorted_items):
        is_last_item = i == len(sorted_items) - 1
        current_prefix = prefix + ("└── " if is_last_item else "├── ")

        # Get the name from the path
        path = item.get("path", "")
        if not path:
            continue

        # Get just the last part of the path for display
        name = path.split("/")[-1]

        # Add the current item
        lines.append(f"{indent}{current_prefix}{name}")

        # If it's a directory, recursively process its contents
        if item.get("type") == "tree":
            # Find all items that are children of this directory
            children = [
                child for child in tree_items
                if child.get("path", "").startswith(path + "/")
                and len(child.get("path", "").split("/")) == len(path.split("/")) + 1
            ]

            if children:
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                new_indent = indent + ("    " if is_last_item else "│   ")
                subtree = draw_github_file_tree(children, new_indent, is_last_item, new_prefix)
                lines.extend(subtree)

    return lines


async def save_github_tree_to_file(repo_url: str, version: str, tree_items: List[Dict], log_file: str = r"logs/repository_trees.log"):
    """
    Saves the repository tree structure to a log file.

    This function:
    - Generates tree structure using draw_file_tree
    - Adds metadata (repo URL, version, timestamp)
    - Appends to existing log file
    - Handles file encoding and errors

    Args:
        repo_url (str): Repository URL
        version (str): Version/branch name
        tree_items (List[Dict]): Tree structure to save
        log_file (str): Path to log file
            Default: "repository_trees.log"
    """
    logger.info(f"Saving tree structure for {repo_url} at version {version}")

    # Create the tree structure
    tree_lines = draw_github_file_tree(tree_items)

    # Format the output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = [
        f"\n{'='*80}",
        f"Repository: {repo_url}",
        f"Version: {version}",
        f"Timestamp: {timestamp}",
        f"{'='*80}\n",
        *tree_lines,
        "\n"
    ]

    # Write to file
    try:
        async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
            await f.write("\n".join(output))
        logger.debug(f"Tree structure saved to {log_file}")
    except Exception as e:
        logger.error(f"Failed to save tree structure: {str(e)}")


def get_file_content(api: GitHubAPI, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
    """
    Retrieves the content of a file from GitHub.

    This function:
    - Converts GitHub web URLs to raw content URLs
    - Handles different file encodings
    - Implements error handling
    - Supports different reference types (branch/tag/commit)

    Args:
        api (GitHubAPI): GitHub API client
        owner (str): Repository owner
        repo (str): Repository name
        path (str): File path relative to repository root
        ref (str): Reference (branch/tag/commit)

    Returns:
        Optional[str]: File content if found, None otherwise
            Content is returned as a string with proper encoding
    """
    try:
        # Convert GitHub web URL to raw content URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Failed to get content for {path}: {str(e)}")
        return None





async def get_github_last_update_time(api: GitHubAPI, owner: str, repo: str, ref: str) -> str:
    """
    Gets the last update time for a repository reference.

    This function:
    - Fetches commit history
    - Extracts commit date
    - Handles different date formats
    - Provides fallback to current year

    Args:
        api (GitHubAPI): GitHub API client
        owner (str): Repository owner
        repo (str): Repository name
        ref (str): Reference (branch/tag/commit)

    Returns:
        str: Year of last update
            Example: "2024"
    """
    try:
        commits_result = await api._make_request(f"/repos/{owner}/{repo}/commits", {"sha": ref, "per_page": 1})
        if isinstance(commits_result, list) and len(commits_result) > 0:
            first_commit = commits_result[0]  # type: ignore
            if isinstance(first_commit, dict):
                commit_info = first_commit.get("commit", {})
                if isinstance(commit_info, dict):
                    author_info = commit_info.get("author", {})
                    if isinstance(author_info, dict):
                        commit_date = author_info.get("date", "")
                        if commit_date:
                            return commit_date.split("-")[0]
    except Exception as e:
        logger.warning(f"Failed to get last update time: {str(e)}")
    return str(datetime.now().year)


async def process_github_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"],
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processes a GitHub repository to extract license information.
    Only analyzes README if all other methods fail to find license info.
    """
    substep_logger = logging.getLogger('substep')
    url_logger = logging.getLogger('url')
    substep_logger.info(f"Starting repository processing: {github_url} (version: {version})")
    if isinstance(github_url, str) and (github_url.startswith("https://www.npmjs.com/") or 
    github_url.startswith("https://registry.npmjs.org/") or 
    github_url.startswith("https://registry.npmmirror.com/") or
    (github_url.startswith("https://mirrors.tencent.com/npm/") and ("/-/" in github_url or "/package/" in github_url))
):
        substep_logger.info(f"Detected npm registry URL: {github_url}, calling process_npm_repository()")
        npm_result = await process_npm_repository(github_url, version)
        if npm_result is not None and npm_result.get("status") != "error":
            return npm_result
        else:
            substep_logger.info("process_npm_repository returned None, continue with GitHub logic.")

    if isinstance(github_url, str) and github_url.startswith("https://pypi.org/"):
        substep_logger.info(f"Detected PyPI registry URL: {github_url}, calling process_pypi_repository()")
        pypi_result = await process_pypi_repository(github_url, version)
        if pypi_result is not None and pypi_result.get("status") != "error":
            return pypi_result
        else:
            substep_logger.info("process_pypi_repository返回 None，继续 GitHub 逻辑。")

    try:
        input_url = github_url

        # Step 1: Check if the URL is a GitHub URL
        substep_logger.info("Step 1/15: Checking if URL is a GitHub URL")
        parsed = urlparse(github_url)
        if parsed.netloc != "github.com":
            substep_logger.info(f"Non-GitHub URL detected: {github_url}")

            # 新增：先判断是否为 NuGet 包
            nuget_result = None
            if name and version:
                try:
                    exists = await check_if_nuget_package_exists(name, version)
                    if exists:
                        substep_logger.info(f"NuGet 包存在，调用 process_nuget_packages")
                        nuget_result = await process_nuget_packages(name, version)
                    else:
                        substep_logger.info(f"不是 NuGet 包或未找到对应版本：{name} {version}")
                except Exception as e:
                    substep_logger.warning(f"NuGet API 调用失败: {e}")

            # 只要 license_type 不为 None 就认为成功并返回
            if nuget_result and nuget_result.get("license_type") is not None:
                substep_logger.info(f"NuGet API 成功，返回 NuGet 结果")
                return nuget_result

            github_url = await find_github_url_from_package_url(github_url, name) or github_url
            if not github_url:
                substep_logger.warning(f"Could not find GitHub URL for {github_url}")
                return {
                    "input_url": input_url,
                    "repo_url": github_url,
                    "input_version": version,
                    "resolved_version": None,
                    "used_default_branch": False,
                    "component_name": None,
                    "license_files": "",
                    "license_analysis": None,
                    "license_type": None,
                    "has_license_conflict": False,
                    "readme_license": None,
                    "license_file_license": None,
                    "copyright_notice": None,
                    "status": "skipped",
                    "license_determination_reason": "Not a GitHub repository and could not find corresponding GitHub URL"
                }
            substep_logger.info(f"Found corresponding GitHub URL: {github_url}")

        # Step 2: Parse URL
        substep_logger.info("Step 2/15: Parsing GitHub URL")
        repo_url, sub_path, kind = parse_github_url(github_url)
        owner, repo = repo_url.split("/")[-2:]
        substep_logger.info(f"Parsed URL: owner={owner}, repo={repo}, kind={kind}, sub_path={sub_path}")

        # Step 3: Get repository info
        substep_logger.info("Step 3/15: Getting repository information")
        try:
            repo_info = await api.get_repo_info(owner, repo)
        except Exception as e:
            substep_logger.warning(f"Error getting repo_info for {owner}/{repo}: {e}, will try with repo=name if name is provided.")
            if name:
                try:
                    repo_info = await api.get_repo_info(owner, name)
                    repo = name
                    substep_logger.info(f"Successfully got repo_info with repo=name: {name}")
                except Exception as e2:
                    substep_logger.error(f"Failed to get repo_info with repo=name: {name}: {e2}")
                    return {
                        "input_url": input_url,
                        "repo_url": repo_url,
                        "input_version": version,
                        "resolved_version": None,
                        "used_default_branch": False,
                        "component_name": None,
                        "license_files": "",
                        "license_analysis": None,
                        "license_type": None,
                        "has_license_conflict": False,
                        "readme_license": None,
                        "license_file_license": None,
                        "copyright_notice": None,
                        "status": "error",
                        "license_determination_reason": f"Failed to get repo_info for both {repo} and {name}"
                    }
            else:
                substep_logger.error(f"No alternative repo name provided, cannot retry get_repo_info.")
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": None,
                    "used_default_branch": False,
                    "component_name": None,
                    "license_files": "",
                    "license_analysis": None,
                    "license_type": None,
                    "has_license_conflict": False,
                    "readme_license": None,
                    "license_file_license": None,
                    "copyright_notice": None,
                    "status": "error",
                    "license_determination_reason": f"Failed to get repo_info for {repo}"
                }
        component_name = repo_info.get("name", repo)
        substep_logger.info(f"Retrieved component name: {component_name}")

        # Step 4: Resolve version
        substep_logger.info("Step 4/15: Resolving version")
        substep_logger.info(f"Resolving version for {owner}/{repo}, requested version: {version}")
        resolved_version, used_default_branch = await resolve_github_version(api, owner, repo, version)
        substep_logger.info(f"Resolved version: {resolved_version} (used_default_branch: {used_default_branch})")

        # Step 5: Try to get license directly from GitHub API
        substep_logger.info("Step 5/15: Checking for license through GitHub API")
        license_info = None
        license_file_analysis = None
        license_url = ""
        license_content = ""
        try:
            license_info = await api.get_license(owner, repo, ref=resolved_version)
            if license_info:
                substep_logger.info("Found license through GitHub API")
                url_logger.info("Found license through GitHub API")
                url_logger.info(f"License info: {json.dumps(license_info, indent=2)}")
                license_content = license_info.get("content", "")
                if license_content:
                    substep_logger.info("Analyzing license content")
                    # 解码Base64编码的内容
                    try:
                        import base64
                        # 检查content是否是Base64编码的
                        if license_info.get("encoding") == "base64":
                            license_content = base64.b64decode(license_content).decode('utf-8')
                            substep_logger.info("Successfully decoded Base64 license content")
                    except Exception as e:
                        substep_logger.warning(f"Failed to decode license content: {str(e)}")
                        # 如果解码失败，继续使用原始内容
                    license_url = license_info.get("_links", {}).get("html") or license_info.get("download_url", "")
                    license_file_analysis = await analyze_license_content_async(license_content, license_url)
                    # 如果LLM分析成功，直接返回结果
                    if license_file_analysis and license_file_analysis.get("licenses"):
                        determination_reason = f"License determined via GitHub API and LLM analysis"
                        copyright_notice = await construct_copyright_notice_async(
                            await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
                            None, license_content
                        )
                        return {
                            "input_url": input_url,
                            "repo_url": repo_url,
                            "input_version": version,
                            "resolved_version": resolved_version,
                            "used_default_branch": used_default_branch,
                            "component_name": component_name,
                            "license_type": license_file_analysis.get("spdx_expression") or (license_file_analysis["licenses"][0] if license_file_analysis["licenses"] else None),
                            "license_files": license_url,
                            "license_analysis": license_file_analysis,
                            "has_license_conflict": False,
                            "readme_license": None,
                            "license_file_license": license_file_analysis.get("spdx_expression") or (license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None),
                            "copyright_notice": copyright_notice,
                            "status": "success",
                            "license_determination_reason": determination_reason
                        }
        except Exception as e:
            substep_logger.warning(f"Error fetching license through GitHub API: {str(e)}")

        # Step 6: Search in repository tree (每次都执行)
        substep_logger.info("Step 6/15: Searching for license in repository tree")
        url_logger.info("Searching in tree for license and thirdparty info")

        tree_response = await api.get_tree(owner, repo, resolved_version)
        tree = tree_response.get("tree", []) if isinstance(tree_response, dict) else []
        substep_logger.info(f"Retrieved tree with {len(tree)} items")

        # 查找 thirdparty 目录
        thirdparty_dirs = find_top_level_thirdparty_dirs(tree)

        if license_file_analysis is None:
            license_file_analysis = {}
        license_file_analysis["thirdparty_dirs"] = thirdparty_dirs
        substep_logger.info(f"found thirdparty_dirs at {thirdparty_dirs}")

        # Step 7: Save tree structure
        #substep_logger.info("Step 7/15: Saving tree structure")
        #await save_github_tree_to_file(repo_url, resolved_version, tree)

        # Find and analyze README（在所有情况下都处理README）
        readme_path = find_readme(tree, sub_path)
        if not readme_path:
            substep_logger.info("No README found in subpath, checking repository root")
            readme_path = find_readme(tree)

        readme_content = None
        readme_license_analysis = None  # 初始化为None
        if readme_path:
            substep_logger.info(f"Found README at: {readme_path}")
            readme_content = await api.get_file_content(owner, repo, readme_path, resolved_version)
            if readme_content:
                substep_logger.info("Analyzing README content for license information")
                # 解码Base64编码的内容（如果需要的话）
                try:
                    # 检查是否是Base64编码的内容
                    if isinstance(readme_content, str) and len(readme_content) > 0:
                        # 尝试解码Base64，如果失败则使用原始内容
                        import base64
                        # 检查是否可能是Base64编码的内容
                        if len(readme_content) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in readme_content):
                            try:
                                readme_content = base64.b64decode(readme_content).decode('utf-8')
                            except Exception:
                                pass  # 如果解码失败，继续使用原始内容
                except Exception as e:
                    substep_logger.warning(f"Failed to decode README content: {str(e)}")
                    # 如果解码失败，继续使用原始内容
                # 构建README文件的GitHub URL
                readme_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{readme_path}"
                readme_license_analysis = await analyze_license_content_async(readme_content, readme_url)
                if readme_license_analysis is not None and "thirdparty_dirs" not in readme_license_analysis:
                    readme_license_analysis["thirdparty_dirs"] = thirdparty_dirs

        # Step 9: Search for license files
        substep_logger.info("Step 9/15: Searching for license files")
        path_map = {
            "tree": tree,
            "resolved_version": resolved_version
        }
        license_files_detailed = find_license_files_detailed(path_map, sub_path, license_keywords)
        substep_logger.info(f"Found {len(license_files_detailed)} license files in subpath")

        # Step 10: Get license content for copyright analysis with intelligent selection
        substep_logger.info("Step 10/15: Getting license content for copyright analysis")
        license_content = None
        license_file_analysis_result = None
        selected_license_file = None
        
        if license_files_detailed:
            # 使用LLM选择主要的license文件
            selected_license_file = await select_primary_license_file(license_files_detailed)
            
            if selected_license_file:
                substep_logger.info(f"Selected primary license file: {selected_license_file['path']}")
                license_content = await api.get_file_content(owner, repo, selected_license_file['path'], resolved_version)
                if license_content:
                    # 使用完整的GitHub URL
                    license_file_analysis_result = await analyze_license_content_async(license_content, selected_license_file['url'])
                        
            if license_file_analysis_result is not None and "thirdparty_dirs" not in license_file_analysis_result:
                license_file_analysis_result["thirdparty_dirs"] = thirdparty_dirs
            # 只要分析有结果，直接返回
            if license_file_analysis_result and license_file_analysis_result.get("licenses"):
                determination_reason = f"Found license files in {sub_path or 'root'}, selected primary: {selected_license_file['filename'] if selected_license_file else ''}"
                copyright_notice = await construct_copyright_notice_async(
                    await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
                    None, license_content
                )
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": resolved_version,
                    "used_default_branch": used_default_branch,
                    "component_name": component_name,
                    "license_files": selected_license_file['url'] if selected_license_file else '',  # 返回选中的主要license文件URL
                    "license_analysis": license_file_analysis_result,
                    "license_type": license_file_analysis_result.get("spdx_expression") or (license_file_analysis_result["licenses"][0] if license_file_analysis_result and license_file_analysis_result["licenses"] else None),
                    "has_license_conflict": False,
                    "readme_license": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None and readme_license_analysis.get("spdx_expression") else None,
                    "license_file_license": license_file_analysis_result.get("spdx_expression") or (license_file_analysis_result["licenses"][0] if license_file_analysis_result and license_file_analysis_result["licenses"] else None),
                    "copyright_notice": copyright_notice,
                    "status": "success",
                    "license_determination_reason": determination_reason
                }

        # Step 13: Try repo-level license
        substep_logger.info("Step 13/15: Trying repo-level license")
        if license_info and license_info.get("license", {}).get("spdx_id"):
            license_type = license_info.get("license", {}).get("spdx_id")
            determination_reason = f"License determined via GitHub API: {license_type}"
            copyright_notice = await construct_copyright_notice_async(
                await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
                None, license_content
            )
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "license_type": license_type,
                "license_files": license_url,
                "license_analysis": license_file_analysis,
                "has_license_conflict": False,
                "readme_license": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None and readme_license_analysis.get("spdx_expression") else None,
                "license_file_license": license_type,  # 添加这行以确保license_file_license被设置
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": determination_reason
            }

        # Step 14: Search entire repo
        substep_logger.info("Step 14/15: Searching entire repository for licenses")
        if not license_files_detailed:
            license_files_detailed = find_license_files_detailed(path_map, "", license_keywords)
            substep_logger.info(f"Found {len(license_files_detailed)} license files in entire repository")

            if license_files_detailed:
                # 使用LLM选择主要的license文件
                selected_license_file = await select_primary_license_file(license_files_detailed)
                
                if selected_license_file:
                    substep_logger.info(f"Selected primary license file: {selected_license_file['path']}")
                    determination_reason = f"Found license files in repository root, selected primary: {selected_license_file['filename']}"
                    license_file_analysis = None
                    license_content = await api.get_file_content(owner, repo, selected_license_file['path'], resolved_version)
                    if license_content:
                        # 解码Base64编码的内容（如果需要的话）
                        try:
                            # 检查是否是Base64编码的内容
                            if isinstance(license_content, str) and len(license_content) > 0:
                                # 尝试解码Base64，如果失败则使用原始内容
                                import base64
                                # 检查是否可能是Base64编码的内容
                                if len(license_content) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in license_content):
                                    try:
                                        license_content = base64.b64decode(license_content).decode('utf-8')
                                    except Exception:
                                        pass  # 如果解码失败，继续使用原始内容
                        except Exception as e:
                            substep_logger.warning(f"Failed to decode license content: {str(e)}")
                            # 如果解码失败，继续使用原始内容
                        license_file_analysis = await analyze_license_content_async(license_content, selected_license_file['url'])
                        
                    if license_file_analysis is not None and "thirdparty_dirs" not in license_file_analysis:
                        license_file_analysis["thirdparty_dirs"] = thirdparty_dirs
                        
                    if license_file_analysis and license_file_analysis.get("licenses"):
                        copyright_notice = await construct_copyright_notice_async(
                            await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
                            None, license_content
                        )
                        return {
                            "input_url": input_url,
                            "repo_url": repo_url,
                            "input_version": version,
                            "resolved_version": resolved_version,
                            "used_default_branch": used_default_branch,
                            "component_name": component_name,
                            "license_files": selected_license_file['url'],  # 返回选中的主要license文件URL
                            "license_analysis": license_file_analysis,
                            "license_type": license_file_analysis.get("spdx_expression") or (license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None),
                            "has_license_conflict": False,
                            "readme_license": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None and readme_license_analysis.get("spdx_expression") else None,
                            "license_file_license": license_file_analysis.get("spdx_expression") or (license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None),
                            "copyright_notice": copyright_notice,
                            "status": "success",
                            "license_determination_reason": determination_reason
                        }

        # 如果在README中找到了许可证信息，并且之前没有通过其他方式找到主要许可证，则使用README中的信息
        if readme_license_analysis and readme_license_analysis.get("licenses") and not license_file_analysis:
            determination_reason = f"Found license info in README: {readme_license_analysis['licenses']}"
            readme_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{readme_path}"
            copyright_notice = await construct_copyright_notice_async(
                await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
                readme_content, None
            )
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "license_files": readme_url,
                "license_analysis": readme_license_analysis,
                "license_type": readme_license_analysis.get("spdx_expression") or (readme_license_analysis["licenses"][0] if readme_license_analysis["licenses"] else None),
                "has_license_conflict": False,
                "readme_license": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None and readme_license_analysis.get("spdx_expression") else None,
                "license_file_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis.get("licenses") else None,  # 从README中获取的许可证
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": determination_reason
            }

        # 如果所有方法都没有找到许可证信息，则返回README分析结果（如果有的话）
        determination_reason = "No license files found in repository"
        readme_url = ""
        if readme_path:
            readme_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{readme_path}"
        license_files_value = ""
        if readme_license_analysis and readme_license_analysis.get("licenses"):
            license_files_value = readme_url
            determination_reason = f"Found license info in README: {readme_license_analysis['licenses']}"
        if readme_license_analysis is not None and "thirdparty_dirs" not in readme_license_analysis:
            readme_license_analysis["thirdparty_dirs"] = thirdparty_dirs
        copyright_notice = await construct_copyright_notice_async(
            await get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
            readme_content, None
        )
        return {
            "input_url": github_url,
            "repo_url": github_url,
            "input_version": version,
            "resolved_version": resolved_version,
            "used_default_branch": used_default_branch,
            "component_name": component_name,
            "license_files": license_files_value,
            "license_analysis": readme_license_analysis,
            "license_type": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None else None or (readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis.get("licenses") else None),
            "has_license_conflict": False,
            "readme_license": readme_license_analysis.get("spdx_expression") if readme_license_analysis is not None and readme_license_analysis.get("spdx_expression") else None,
            "license_file_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis.get("licenses") else None,  # 从README中获取的许可证
            "copyright_notice": copyright_notice,
            "status": "success",
            "license_determination_reason": determination_reason
        }

    except Exception as e:
        error_msg = str(e)
        substep_logger.error(f"Error processing repository {github_url}: {error_msg}", exc_info=True)
        return {
            "input_url": github_url,
            "repo_url": github_url,
            "input_version": version,
            "resolved_version": None,
            "used_default_branch": False,
            "component_name": None,
            "error": error_msg,
            "status": "error",
            "license_determination_reason": f"Error: {error_msg}"
        }

def deduplicate_license_files(license_files: List[str], owner: str, repo: str, resolved_version: str) -> List[str]:
    normalized = set()
    result = []
    for license_file in license_files:
        # 如果已经是 GitHub web 路径，直接加入
        if license_file.startswith("https://github.com/"):
            norm_key = license_file
            web_url = license_file
        elif license_file.startswith("https://raw.githubusercontent.com"):
            parts = license_file.replace("https://raw.githubusercontent.com/", "").split("/")
            if len(parts) >= 4:
                owner_, repo_, ref, *path_parts = parts
                path = "/".join(path_parts)
                norm_key = (owner_, repo_, path)
                web_url = f"https://github.com/{owner_}/{repo_}/blob/{ref}/{path}"
            else:
                norm_key = license_file
                web_url = license_file
        else:
            norm_key = (owner, repo, license_file.replace("\\", "/"))
            web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{license_file}"
        if norm_key not in normalized:
            normalized.add(norm_key)
            result.append(web_url)
    return result

# async def find_github_url_from_package_url(package_url: str, name: Optional[str] = None) -> Optional[str]:
#     """
#     异步：根据 package_url 和 name，调用 LLM 查找 GitHub 仓库链接
#     """
#     if not USE_LLM:
#         logger.info("LLM is disabled, skipping GitHub URL lookup")
#         return None

#     try:
#         prompt = PROMPTS["github_url_finder"].format(package_url=package_url, name=name or "")
#         llm_logger.info("GitHub URL Lookup Request:")
#         llm_logger.info(f"Prompt: {prompt}")

#         model = genai.GenerativeModel(GEMINI_CONFIG["model"])
#         response = await model.generate_content(prompt)  # 异步调用

#         llm_logger.info("GitHub URL Lookup Response:")
#         llm_logger.info(f"Response: {response.text}")

#         if response.text:
#             json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#                 github_url = result.get("github_url")
#                 confidence = result.get("confidence", 0.0)
#                 llm_logger.info(f"Found GitHub URL: {github_url} with confidence {confidence}")
#                 if github_url and confidence >= 0.7:
#                     return github_url
#                 else:
#                     llm_logger.info(f"No confident GitHub URL match found (confidence: {confidence})")
#             else:
#                 llm_logger.warning("No JSON found in GitHub URL lookup response")
#     except Exception as e:
#         llm_logger.error(f"Failed to find GitHub URL: {str(e)}", exc_info=True)
#     return None




async def find_github_url_from_package_url(package_url: str, name: Optional[str] = None) -> Optional[str]:
    """
    使用 LLM 提供商，根据 package_url 和 name 异步查找 GitHub 仓库链接
    """
    if not USE_LLM:
        logger.info("LLM is disabled, skipping GitHub URL lookup")
        return None

    prompt = PROMPTS["github_url_finder"].format(package_url=package_url, name=name or "")
    llm_logger.info("GitHub URL Lookup Request:")
    llm_logger.info(f"Prompt: {prompt}")

    provider = get_llm_provider()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        response = await provider.generate_async(prompt)
        llm_logger.info("GitHub URL Lookup Response:")
        llm_logger.info(f"Response: {response}")

        if response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                github_url = result.get("github_url")
                confidence = result.get("confidence", 0.0)
                llm_logger.info(f"Found GitHub URL: {github_url} with confidence {confidence}")
                if github_url and confidence >= 0.7:
                    return github_url
                else:
                    llm_logger.info(f"No confident GitHub URL match found (confidence: {confidence})")
            else:
                llm_logger.warning("No JSON found in GitHub URL lookup response")
    except Exception as e:
        llm_logger.error(f"Failed to find GitHub URL: {str(e)}", exc_info=True)
    return None