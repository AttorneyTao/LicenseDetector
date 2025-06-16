
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

from core.utils import is_sha_version


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
    
    def __init__(self):

        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.error("GITHUB_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN environment variable not set")
        
        logger.info("Initializing GitHub API client")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        # Configure proxy settings
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy:
            logger.info(f"Using proxy: {proxy}")
            self.session = requests.Session()
            self.session.proxies = {
                "http": proxy,
                "https": proxy
            }
        else:
            logger.info("No proxy configured")
            self.session = requests.Session()
            
        self.session.headers.update(self.headers)
        
        # Test connection
        try:
            logger.info("Testing GitHub API connection...")
            self._make_request("/rate_limit")
            logger.info("GitHub API connection successful")
        except requests.exceptions.ProxyError as e:
            logger.error(f"Proxy connection failed: {str(e)}")
            logger.info("Retrying without proxy...")
            # Retry without proxy
            self.session.proxies = {}
            self._make_request("/rate_limit")
            logger.info("GitHub API connection successful without proxy")
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
        per_page = 100  # GitHub API max per_page is 100
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


def resolve_github_version(api: GitHubAPI, owner: str, repo: str, version: Optional[str]) -> Tuple[str, bool]:
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
    repo_info = api.get_repo_info(owner, repo)
    default_branch = repo_info["default_branch"]
    version_resolve_logger.info(f"Repository default branch: {default_branch}")

    # Get all branches and tags
    branches = api.get_branches(owner, repo)
    tags = api.get_tags(owner, repo)
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