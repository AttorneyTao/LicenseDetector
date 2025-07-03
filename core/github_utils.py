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

from core.npm_utils import process_npm_repository
from core.utils import analyze_license_content, construct_copyright_notice, find_license_files, find_readme, is_sha_version, analyze_license_content_async, construct_copyright_notice_async



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


def save_github_tree_to_file(repo_url: str, version: str, tree_items: List[Dict], log_file: str = r"logs/repository_trees.log"):
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
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(output))
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


def get_github_last_update_time(api: GitHubAPI, owner: str, repo: str, ref: str) -> str:
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
        # Get the commit history for the ref
        commits = api._make_request(f"/repos/{owner}/{repo}/commits", {"sha": ref, "per_page": 1})
        if commits and len(commits) > 0:
            # Get the commit date from the first (most recent) commit
            commit_date = commits[0]["commit"]["author"]["date"]
            # Extract the year from the date
            return commit_date.split("-")[0]
    except Exception as e:
        logger.warning(f"Failed to get last update time: {str(e)}")
    return datetime.now().year


async def process_github_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"],
    name: str = None
) -> Dict[str, Any]:
    """
    Processes a GitHub repository to extract license information.

    This function implements a comprehensive analysis pipeline:
    1. URL validation and parsing
    2. Repository information retrieval
    3. Version resolution
    4. License file detection
    5. License content analysis
    6. Copyright notice extraction
    7. Conflict detection

    Args:
        api (GitHubAPI): GitHub API client
        github_url (str): GitHub repository URL
        version (Optional[str]): Version to analyze
        license_keywords (List[str]): Keywords to identify license files

    Returns:
        Dict[str, Any]: Processing results including:
            - input_url: Original input URL
            - repo_url: Repository URL
            - input_version: Requested version
            - resolved_version: Actual version used
            - used_default_branch: Whether default branch was used
            - component_name: Name of the component
            - license_files: URLs to license files
            - license_analysis: Detailed license analysis
            - license_type: Primary license type
            - has_license_conflict: Whether conflicts were found
            - readme_license: License from README
            - license_file_license: License from files
            - copyright_notice: Copyright information
            - status: Processing status
            - license_determination_reason: Explanation of results
    """
    substep_logger = logging.getLogger('substep')
    url_logger = logging.getLogger('url')
    substep_logger.info(f"Starting repository processing: {github_url} (version: {version})")
    if isinstance(github_url, str) and (github_url.startswith("https://www.npmjs.com/") or github_url.startswith("https://registry.npmjs.org/")):
        substep_logger.info(f"Detected npm registry URL: {github_url}, calling process_npm_repository()")
        npm_result = await process_npm_repository(github_url, version)
        if npm_result is not None and npm_result.get("status") != "error":
            return npm_result
        else:
            substep_logger.info("process_npm_repository returned None, continue with GitHub logic.")

    substep_logger.info(f"Starting repository processing: {github_url} (version: {version})")
    try:
        # Store original input URL
        input_url = github_url


        # Step 1: Check if the URL is a GitHub URL
        substep_logger.info("Step 1/15: Checking if URL is a GitHub URL")
        parsed = urlparse(github_url)
        if parsed.netloc != "github.com":
            substep_logger.info(f"Non-GitHub URL detected: {github_url}")
            github_url = find_github_url_from_package_url(github_url)
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
            repo_info = api.get_repo_info(owner, repo)
        except Exception as e:
            substep_logger.warning(f"Error getting repo_info for {owner}/{repo}: {e}, will try with repo=name if name is provided.")
            if name:
                try:
                    repo_info = api.get_repo_info(owner, name)
                    repo = name  # 更新repo变量为name
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
        resolved_version, used_default_branch = resolve_github_version(api, owner, repo, version)
        substep_logger.info(f"Resolved version to: {resolved_version}, used_default_branch: {used_default_branch}")

        # Step 5: Try to get license directly from GitHub API
        substep_logger.info("Step 5/15: Checking for license through GitHub API")
        try:
            license_info = api.get_license(owner, repo, ref=resolved_version)
            if license_info:
                substep_logger.info("Found license through GitHub API")
                url_logger.info("Found license through GitHub API")
                url_logger.info(f"License info: {json.dumps(license_info, indent=2)}")

                # Get license content and analyze it
                license_content = license_info.get("content", "")
                if license_content:
                    substep_logger.info("Analyzing license content")
                    license_file_analysis = await analyze_license_content_async(license_content)

                    # Use the html_url from _links if available
                    license_url = license_info.get("_links", {}).get("html")
                    if not license_url:
                        # Fallback to constructing from download_url
                        license_url = license_info.get("download_url", "")
                        if license_url.startswith("https://raw.githubusercontent.com"):
                            parts = license_url.replace("https://raw.githubusercontent.com/", "").split("/")
                            if len(parts) >= 3:
                                owner, repo, *path_parts = parts
                                path = "/".join(path_parts)
                                license_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{path}"

                    url_logger.info(f"Using license URL: {license_url}")
                    substep_logger.info("Step 5/15 completed successfully")

                    return {
                        "input_url": input_url,
                        "repo_url": repo_url,
                        "input_version": version,
                        "resolved_version": resolved_version,
                        "used_default_branch": used_default_branch,
                        "component_name": component_name,
                        "license_files": license_url,
                        "license_analysis": license_file_analysis,
                        "license_type": license_info.get("license", {}).get("spdx_id"),
                        "has_license_conflict": False,
                        "readme_license": None,
                        "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                        "copyright_notice": await construct_copyright_notice_async(get_github_last_update_time(api,owner,repo,resolved_version), owner, repo, resolved_version, component_name, None, license_content),
                        "status": "success",
                        "license_determination_reason": "License found through GitHub API"
                    }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                substep_logger.warning(f"No license found through GitHub API for {owner}/{repo}")
                url_logger.info(f"No license found through GitHub API for {owner}/{repo}")
            else:
                substep_logger.error(f"Error fetching license through GitHub API: {str(e)}")
                url_logger.error(f"Error fetching license through GitHub API: {str(e)}")

        # Step 6: Search in repository tree
        substep_logger.info("Step 6/15: Searching for license in repository tree")
        url_logger.info("No license found through GitHub API, searching in tree")

        # Get tree using the resolved version
        substep_logger.info("Getting repository tree")
        tree_response = api.get_tree(owner, repo, resolved_version)
        if not isinstance(tree_response, dict):
            substep_logger.error(f"Invalid tree response: {tree_response}")
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "error": "Invalid tree response from GitHub API",
                "status": "error",
                "license_determination_reason": "Error: Invalid tree response"
            }

        tree = tree_response.get("tree", [])
        if not isinstance(tree, list):
            substep_logger.error(f"Invalid tree data: {tree}")
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "error": "Invalid tree data from GitHub API",
                "status": "error",
                "license_determination_reason": "Error: Invalid tree data"
            }

        substep_logger.info(f"Retrieved tree with {len(tree)} items")

        # Step 7: Save tree structure
        substep_logger.info("Step 7/15: Saving tree structure")
        save_github_tree_to_file(repo_url, resolved_version, tree)

        # Step 8: Find and analyze README
        substep_logger.info("Step 8/15: Looking for README file")
        readme_path = find_readme(tree, sub_path)
        if not readme_path:
            substep_logger.info("No README found in subpath, checking repository root")
            readme_path = find_readme(tree)

        readme_content = None
        readme_license_analysis = None
        if readme_path:
            substep_logger.info(f"Found README at: {readme_path}")
            readme_content = get_file_content(api, owner, repo, readme_path, resolved_version)
            if readme_content:
                substep_logger.info("Analyzing README content for license information")
                readme_license_analysis = await analyze_license_content_async(readme_content)
                if readme_license_analysis["licenses"]:
                    substep_logger.info(f"Found license information in README: {readme_license_analysis}")

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
        if license_files:
            for license_file in license_files:
                license_content = get_file_content(api, owner, repo, license_file.split("/")[-1], resolved_version)
                if license_content:
                    break

        # Step 11: Get copyright notice
        substep_logger.info("Step 11/15: Constructing copyright notice")
        copyright_notice = await construct_copyright_notice_async(
            get_github_last_update_time(api, owner, repo, resolved_version), owner, repo, resolved_version, component_name,
            readme_content, license_content
        )
        substep_logger.info(f"Copyright notice: {copyright_notice}")

        # Step 12: Analyze found licenses
        substep_logger.info("Step 12/15: Analyzing found licenses")
        if license_files:
            license_paths = [url.split("/")[-1] for url in license_files]
            determination_reason = f"Found license files in {sub_path or 'root'}: {', '.join(license_paths)}"
            substep_logger.info(determination_reason)

            # If no license info found in README, analyze license files
            license_file_analysis = None
            if not readme_license_analysis or not readme_license_analysis["licenses"]:
                substep_logger.info("No license info in README, analyzing license files")
                for license_file in license_files:
                    license_content = get_file_content(api, owner, repo, license_file.split("/")[-1], resolved_version)
                    if license_content:
                        license_file_analysis = await analyze_license_content_async(license_content)
                        if license_file_analysis["licenses"]:
                            break

            # Check for license conflicts
            license_conflict = False
            if readme_license_analysis and license_file_analysis:
                readme_licenses = set(readme_license_analysis["licenses"])
                file_licenses = set(license_file_analysis["licenses"])
                if readme_licenses != file_licenses:
                    license_conflict = True
                    substep_logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")

            # Use license file analysis if available, otherwise use README analysis
            final_analysis = license_file_analysis or readme_license_analysis

            # Convert raw GitHub URLs to web URLs
            substep_logger.info("Converting license URLs to web URLs")
            web_license_files = []
            for license_file in license_files:
                if license_file.startswith("https://raw.githubusercontent.com"):
                    # Convert raw URL to web URL
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

            substep_logger.info("Step 12/15 completed successfully")
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "license_files": "\n".join(web_license_files),
                "license_analysis": final_analysis,
                "license_type": final_analysis["licenses"][0] if final_analysis and final_analysis["licenses"] else None,
                "has_license_conflict": license_conflict,
                "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
                "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": determination_reason
            }

        # Step 13: Try repo-level license
        substep_logger.info("Step 13/15: Trying repo-level license")
        try:
            license_info = api.get_license(owner, repo, ref=resolved_version)
            if license_info:
                license_type = license_info.get("license", {}).get("spdx_id")
                determination_reason = f"License determined via GitHub API: {license_type}"
                substep_logger.info(determination_reason)

                # If no license info found in README, analyze the license content
                license_file_analysis = None
                if not readme_license_analysis or not readme_license_analysis["licenses"]:
                    substep_logger.info("Analyzing repo-level license content")
                    license_content = get_file_content(api, owner, repo, "LICENSE", resolved_version)
                    if license_content:
                        license_file_analysis = await analyze_license_content_async(license_content)

                # Check for license conflicts
                license_conflict = False
                if readme_license_analysis and license_file_analysis:
                    readme_licenses = set(readme_license_analysis["licenses"])
                    file_licenses = set(license_file_analysis["licenses"])
                    if readme_licenses != file_licenses:
                        license_conflict = True
                        substep_logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")

                # Use license file analysis if available, otherwise use README analysis
                final_analysis = license_file_analysis or readme_license_analysis

                # Convert raw GitHub URL to web URL
                license_url = license_info["download_url"]
                if license_url.startswith("https://raw.githubusercontent.com"):
                    parts = license_url.replace("https://raw.githubusercontent.com/", "").split("/")
                    if len(parts) >= 3:
                        owner, repo, *path_parts = parts
                        path = "/".join(path_parts)
                        web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{path}"
                        license_url = web_url

                substep_logger.info("Step 13/15 completed successfully")
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": resolved_version,
                    "used_default_branch": used_default_branch,
                    "component_name": component_name,
                    "license_type": license_type,
                    "license_files": license_url,
                    "license_analysis": final_analysis,
                    "has_license_conflict": license_conflict,
                    "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
                    "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                    "copyright_notice": copyright_notice,
                    "status": "success",
                    "license_determination_reason": determination_reason
                }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                substep_logger.warning(f"No license found for {owner}/{repo} at version {resolved_version}")
            else:
                substep_logger.error(f"Error fetching license: {str(e)}")

        # Step 14: Search entire repo
        substep_logger.info("Step 14/15: Searching entire repository for licenses")
        if not license_files:
            license_files = find_license_files(path_map, "", license_keywords)
            substep_logger.info(f"Found {len(license_files)} license files in entire repository")

            if license_files:
                license_paths = [url.split("/")[-1] for url in license_files]
                determination_reason = f"Found license files in repository root: {', '.join(license_paths)}"
                substep_logger.info(determination_reason)

                # If no license info found in README, analyze license files
                license_file_analysis = None
                if not readme_license_analysis or not readme_license_analysis["licenses"]:
                    substep_logger.info("Analyzing found license files")
                    for license_file in license_files:
                        license_content = get_file_content(api, owner, repo, license_file.split("/")[-1], resolved_version)
                        if license_content:
                            license_file_analysis = await analyze_license_content_async(license_content)
                            if license_file_analysis["licenses"]:
                                break

                # Check for license conflicts
                license_conflict = False
                if readme_license_analysis and license_file_analysis:
                    readme_licenses = set(readme_license_analysis["licenses"])
                    file_licenses = set(license_file_analysis["licenses"])
                    if readme_licenses != file_licenses:
                        license_conflict = True
                        substep_logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")

                # Use license file analysis if available, otherwise use README analysis
                final_analysis = license_file_analysis or readme_license_analysis

                # Convert raw GitHub URLs to web URLs
                substep_logger.info("Converting license URLs to web URLs")
                web_license_files = []
                for license_file in license_files:
                    if license_file.startswith("https://raw.githubusercontent.com"):
                        # Convert raw URL to web URL
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

                substep_logger.info("Step 14/15 completed successfully")
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": resolved_version,
                    "used_default_branch": used_default_branch,
                    "component_name": component_name,
                    "license_files": "\n".join(web_license_files),
                    "license_analysis": final_analysis,
                    "license_type": final_analysis["licenses"][0] if final_analysis and final_analysis["licenses"] else None,
                    "has_license_conflict": license_conflict,
                    "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
                    "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                    "copyright_notice": copyright_notice,
                    "status": "success",
                    "license_determination_reason": determination_reason
                }

        # Step 15: No licenses found
        substep_logger.info("Step 15/15: No licenses found in repository")
        determination_reason = "No license files found in repository"
        substep_logger.warning(determination_reason)
                # 新增逻辑：如果readme_license_analysis有license，license_files用readme的完整url
        readme_url = ""
        if readme_path:
            readme_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{readme_path}"
        license_files_value = ""
        if readme_license_analysis and readme_license_analysis.get("licenses"):
            license_files_value = readme_url
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
            "input_url": input_url,
            "repo_url": github_url,
            "input_version": version,
            "resolved_version": None,
            "used_default_branch": False,
            "component_name": None,
            "error": error_msg,
            "status": "error",
            "license_determination_reason": f"Error: {error_msg}"
        }