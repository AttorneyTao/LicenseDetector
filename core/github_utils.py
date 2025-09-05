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
import google.generativeai as genai
import httpx  # 新增：导入 httpx
import asyncio  # 新增：导入 asyncio
from httpx import AsyncClient
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential

from core.npm_utils import process_npm_repository
from core.pypi_utils import process_pypi_repository
from core.utils import analyze_license_content, construct_copyright_notice, find_license_files, find_readme, find_top_level_thirdparty_dirs, is_sha_version, analyze_license_content_async, construct_copyright_notice_async



class Kind(Enum):
    REPO = "REPO"
    DIR = "DIR"
    FILE = "FILE"


#==============================================================================
from .config import GEMINI_CONFIG

load_dotenv()
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

logger = logging.getLogger('main')
llm_logger = logging.getLogger('llm_interaction')

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
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {os.getenv('GITHUB_TOKEN')}"
        }
    
    async def initialize(self):  # 新增异步初始化方法
        """异步初始化和测试连接"""
        try:
            logger.info("Testing GitHub API connection...")
            await self._make_request("/rate_limit")
            logger.info("GitHub API connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to GitHub API: {str(e)}")
            raise

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
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
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        异步请求处理，包含速率限制处理
        """
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Making request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        async with AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url, 
                params=params,
                headers=self.headers
            )
            return response.json()
    async def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        async with AsyncClient() as client:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
            response = await client.get(raw_url)
            return response.text
    def get_repo_info(self, owner: str, repo: str) -> Dict:
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
        return self._make_request(f"/repos/{owner}/{repo}")

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

    def get_branches(self, owner: str, repo: str) -> List[Dict]:
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
        return self._make_request(f"/repos/{owner}/{repo}/branches")

    async def get_branches(self, owner: str, repo: str) -> List[Dict]:
        logger.info(f"Fetching branches for {owner}/{repo}")
        return await self._make_request(f"/repos/{owner}/{repo}/branches")

    def get_tags(self, owner: str, repo: str) -> List[Dict]:
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
        tags = []
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
            tags.extend(response)
            if len(response) < per_page:
                break
            page += 1
        return tags

    async def get_tags(self, owner: str, repo: str) -> List[Dict]:
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
            tags.extend(response)
            if len(response) < per_page:
                break
            page += 1
        return tags

    def get_tree(self, owner: str, repo: str, sha: str) -> Dict:
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
        return self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})

    async def get_tree(self, owner: str, repo: str, sha: str) -> Dict:
        logger.info(f"Fetching tree for {owner}/{repo} at {sha}")
        return await self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})

    def get_license(self, owner: str, repo: str, ref: Optional[str] = None) -> Optional[Dict]:
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
            return self._make_request(f"/repos/{owner}/{repo}/license", params=params)
        except requests.exceptions.HTTPError as e:
            # Handle 404 (no license found) gracefully
            if e.response.status_code == 404:
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
            if getattr(e, 'response', None) and e.response.status_code == 404:
                logger.warning(f"No license found for {owner}/{repo}")
                return None
            logger.error(f"Error fetching license: {str(e)}")
            raise

def find_github_url_from_package_url(package_url: str) -> Optional[str]:
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
        # prompt = f"""
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
        
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        
        llm_logger.info("GitHub URL Lookup Response:")
        llm_logger.info(f"Response: {response.text}")
        
        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
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

    # Get default branch
    repo_info = await api.get_repo_info(owner, repo)  # 添加 await
    # 安全获取 default_branch，没有则 fallback
    default_branch = repo_info.get("default_branch")
    if not default_branch:
        # fallback 顺序：main > master > 第一个 branch > 'main'
        branches = await api.get_branches(owner, repo)
        branch_names = [b.get("name") for b in branches if "name" in b]
        if "main" in branch_names:
            default_branch = "main"
        elif "master" in branch_names:
            default_branch = "master"
        elif branch_names:
            default_branch = branch_names[0]
        else:
            default_branch = "main"
        logging.getLogger('version_resolve').warning(
            f"repo_info for {owner}/{repo} missing 'default_branch', fallback to: {default_branch}"
        )
    version_resolve_logger.info(f"Repository default branch: {default_branch}")

    # Get all branches and tags
    branches = await api.get_branches(owner, repo)  # 添加 await
    tags = await api.get_tags(owner, repo)  # 添加 await，并确保 get_tags 也是异步的
    candidate_versions = [b["name"] for b in branches] + [t["name"] for t in tags]
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

            model = genai.GenerativeModel(GEMINI_CONFIG["model"])
            response = model.generate_content(prompt)
            llm_logger.info("Version Resolve Response:")
            llm_logger.info(f"Response: {response.text}")
            version_resolve_logger.info("Version Resolve LLM Response:")
            version_resolve_logger.info(f"Response: {response.text}")

            if response.text:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
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
            version_resolve_logger.error(f"Failed to resolve version via LLM: {str(e)}", exc_info=True)

    # fallback
    version_resolve_logger.info("No version matched, using default branch")
    return default_branch, True


def parse_github_url(url: str) -> Tuple[str, str, Kind]:
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
        commits = await api._make_request(f"/repos/{owner}/{repo}/commits", {"sha": ref, "per_page": 1})
        if commits and len(commits) > 0:
            commit_date = commits[0]["commit"]["author"]["date"]
            return commit_date.split("-")[0]
    except Exception as e:
        logger.warning(f"Failed to get last update time: {str(e)}")
    return str(datetime.now().year)


async def process_github_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"],
    name: str = None
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
            github_url = await find_github_url_from_package_url(github_url, name)
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
                    license_file_analysis = await analyze_license_content_async(license_content)
                    license_url = license_info.get("_links", {}).get("html") or license_info.get("download_url", "")
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
        substep_logger.info("Step 7/15: Saving tree structure")
        await save_github_tree_to_file(repo_url, resolved_version, tree)

        # Step 9: Search for license files
        substep_logger.info("Step 9/15: Searching for license files")
        path_map = {
            "tree": tree,
            "resolved_version": resolved_version
        }
        license_files = find_license_files(path_map, sub_path, license_keywords)
        substep_logger.info(f"Found {len(license_files)} license files in subpath")

        # Step 10: Get license content for copyright analysis
        substep_logger.info("Step 10/15: Getting license content for copyright analysis")
        license_content = None
        license_file_analysis_result = None
        if license_files:
            for license_file in license_files:
                license_content = await api.get_file_content(owner, repo, license_file.split("/")[-1], resolved_version)
                if license_content:
                    license_file_analysis_result = await analyze_license_content_async(license_content)
                    if license_file_analysis_result and license_file_analysis_result.get("licenses"):
                        break
            if license_file_analysis_result is not None and "thirdparty_dirs" not in license_file_analysis_result:
                license_file_analysis_result["thirdparty_dirs"] = thirdparty_dirs
            # 只要tree分析有结果，直接返回
            if license_file_analysis_result and license_file_analysis_result.get("licenses"):
                web_license_files = deduplicate_license_files(license_files, owner, repo, resolved_version)
                # for license_file in license_files:
                #     if license_file.startswith("https://raw.githubusercontent.com"):
                #         parts = license_file.replace("https://raw.githubusercontent.com/", "").split("/")
                #         if len(parts) >= 3:
                #             owner, repo, *path_parts = parts
                #             path = "/".join(path_parts)
                #             web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{path}"
                #             web_license_files.append(web_url)
                #         else:
                #             web_license_files.append(license_file)
                #     else:
                #         web_license_files.append(license_file)
                determination_reason = f"Found license files in {sub_path or 'root'}: {', '.join([url.split('/')[-1] for url in license_files])}"
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
                    "license_files": "\n".join(web_license_files),
                    "license_analysis": license_file_analysis_result,
                    "license_type": license_file_analysis_result["licenses"][0] if license_file_analysis_result and license_file_analysis_result["licenses"] else None,
                    "has_license_conflict": False,
                    "readme_license": None,
                    "license_file_license": license_file_analysis_result["licenses"][0] if license_file_analysis_result and license_file_analysis_result["licenses"] else None,
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
                "readme_license": None,
                "license_file_license": None,
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": determination_reason
            }

        # Step 14: Search entire repo
        substep_logger.info("Step 14/15: Searching entire repository for licenses")
        if not license_files:
            license_files = find_license_files(path_map, "", license_keywords)
            substep_logger.info(f"Found {len(license_files)} license files in entire repository")

            if license_files:
                license_paths = [url.split("/")[-1] for url in license_files]
                determination_reason = f"Found license files in repository root: {', '.join(license_paths)}"
                license_file_analysis = None
                for license_file in license_files:
                    license_content = await api.get_file_content(owner, repo, license_file.split("/")[-1], resolved_version)
                    if license_content:
                        license_file_analysis = await analyze_license_content_async(license_content)
                        if license_file_analysis and license_file_analysis.get("licenses"):
                            break
                if license_file_analysis is not None and "thirdparty_dirs" not in license_file_analysis:
                    license_file_analysis["thirdparty_dirs"] = thirdparty_dirs
                if license_file_analysis and license_file_analysis.get("licenses"):
                    web_license_files = []
                    for license_file in license_files:
                        if license_file.startswith("https://raw.githubusercontent.com"):
                            parts = license_file.replace("https://raw.githubusercontent.com/", "").split("/")
                            if len(parts) >= 3:
                                owner, repo, *path_parts = parts
                                path = "/".join(path_parts)
                                web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{path}"
                                web_license_files.append(web_url)
                            else:
                                web_license_files.append(license_file)
                        else:
                            web_license_files.append(license_file)
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
                        "license_files": "\n".join(web_license_files),
                        "license_analysis": license_file_analysis,
                        "license_type": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                        "has_license_conflict": False,
                        "readme_license": None,
                        "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                        "copyright_notice": copyright_notice,
                        "status": "success",
                        "license_determination_reason": determination_reason
                    }

        # Step 8: Find and analyze README（兜底，只有前面都没找到license时才执行）
        substep_logger.info("Step 8/15: Looking for README file")
        readme_path = find_readme(tree, sub_path)
        if not readme_path:
            substep_logger.info("No README found in subpath, checking repository root")
            readme_path = find_readme(tree)

        readme_content = None
        readme_license_analysis = None
        if readme_path:
            substep_logger.info(f"Found README at: {readme_path}")
            readme_content = await api.get_file_content(owner, repo, readme_path, resolved_version)
            if readme_content:
                substep_logger.info("Analyzing README content for license information")
                readme_license_analysis = await analyze_license_content_async(readme_content)
                if readme_license_analysis is not None and "thirdparty_dirs" not in readme_license_analysis:
                    readme_license_analysis["thirdparty_dirs"] = thirdparty_dirs
                if readme_license_analysis and readme_license_analysis.get("licenses"):
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
                        "license_type": readme_license_analysis["licenses"][0] if readme_license_analysis["licenses"] else None,
                        "has_license_conflict": False,
                        "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis["licenses"] else None,
                        "license_file_license": None,
                        "copyright_notice": copyright_notice,
                        "status": "success",
                        "license_determination_reason": determination_reason
                    }

        # Step 15: No licenses found
        substep_logger.info("Step 15/15: No licenses found in repository")
        determination_reason = "No license files found in repository"
        substep_logger.warning(determination_reason)
        readme_url = ""
        if readme_path:
            readme_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{readme_path}"
        license_files_value = ""
        if readme_license_analysis and readme_license_analysis.get("licenses"):
            license_files_value = readme_url
        if readme_license_analysis is not None and "thirdparty_dirs" not in readme_license_analysis:
            readme_license_analysis["thirdparty_dirs"] = thirdparty_dirs
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
            "license_files": license_files_value,
            "license_analysis": readme_license_analysis,
            "license_type": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
            "has_license_conflict": False,
            "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
            "license_file_license": None,
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
    """
    对 license_files 去重，确保同一个物理文件只出现一次（raw 路径和本地路径归一化后只保留一个）。
    支持 raw.githubusercontent.com 路径和相对路径。
    """
    normalized = set()
    result = []
    for license_file in license_files:
        if license_file.startswith("https://raw.githubusercontent.com"):
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
            # 相对路径或其他情况
            norm_key = (owner, repo, license_file.replace("\\", "/"))
            web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{license_file}"
        if norm_key not in normalized:
            normalized.add(norm_key)
            result.append(web_url)
    return result

async def find_github_url_from_package_url(package_url: str, name: Optional[str] = None) -> Optional[str]:
    """
    异步：根据 package_url 和 name，调用 LLM 查找 GitHub 仓库链接
    """
    if not USE_LLM:
        logger.info("LLM is disabled, skipping GitHub URL lookup")
        return None

    try:
        prompt = PROMPTS["github_url_finder"].format(package_url=package_url, name=name or "")
        llm_logger.info("GitHub URL Lookup Request:")
        llm_logger.info(f"Prompt: {prompt}")

        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = await model.generate_content(prompt)  # 异步调用

        llm_logger.info("GitHub URL Lookup Response:")
        llm_logger.info(f"Response: {response.text}")

        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
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


