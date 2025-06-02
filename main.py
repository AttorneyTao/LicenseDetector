# Standard library imports
import os  # For environment variables and file operations
import re  # For regular expression operations
import json  # For JSON parsing and formatting
import time  # For rate limit handling and delays
import logging  # For logging functionality
import logging.handlers  # For rotating file handlers
from datetime import datetime  # For timestamp generation
from enum import Enum  # For type enumeration
from typing import Optional, Tuple, List, Dict, Any  # For type hints
from urllib.parse import urlparse, unquote  # For URL parsing

# Third-party imports
import pandas as pd  # For Excel file handling
import requests  # For HTTP requests
from dotenv import load_dotenv  # For loading environment variables
from rapidfuzz import process, fuzz  # For fuzzy string matching
from tqdm import tqdm  # For progress bar
import google.generativeai as genai  # For Gemini API integration

# Constants
SCORE_THRESHOLD = 65  # Minimum similarity score for version matching

# Configure logging
# Set up the main logger with a rotating file handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logging level to INFO for detailed output

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure rotating file handler for main log
# This will create new log files when the current one reaches 5MB
# and keep up to 5 backup files
handler = logging.handlers.RotatingFileHandler(
    "logs/main.log",
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5,
    encoding='utf-8'
)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Set up specialized loggers for different aspects of the application
# URL construction logger for tracking URL processing
url_logger = logging.getLogger("url_construction")
url_logger.setLevel(logging.INFO)
url_handler = logging.FileHandler("logs/url_construction.log", encoding='utf-8')
url_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
url_logger.addHandler(url_handler)

# LLM interaction logger for tracking Gemini API calls
llm_logger = logging.getLogger("llm_interaction")
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler("logs/llm_interaction.log", encoding='utf-8')
llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(llm_handler)

# Configure substep logging
substep_logger = logging.getLogger('substep')  # Logger for detailed step tracking
substep_logger.setLevel(logging.INFO)
substep_handler = logging.FileHandler('substep.log', encoding='utf-8')
substep_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
substep_logger.addHandler(substep_handler)

# Set UTF-8 encoding for stdout and stderr
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Load environment variables
load_dotenv()

# Configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"  # Whether to use LLM analysis
logger.info(f"LLM analysis is {'enabled' if USE_LLM else 'disabled'}")

# Gemini API Configuration
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),  # API key for Gemini
    "model": "gemini-2.5-flash-preview-05-20",  # Model version to use
    "prompts": {
        "license_analysis": """
        Analyze the following text and determine:
        1. What is the main license specified? Use standard SPDX identifiers (e.g., Apache-2.0, MIT, GPL-3.0, etc.)
        2. Is this project dual licensed (e.g., MIT OR Apache-2.0)? If yes, what are the licenses and their relationship?
        3. Are there any mentions of third-party components with different licenses? If yes, where can they be found (e.g., specific section or URL)?
        4. What is the relationship between the main license(s) and third-party licenses?
        
        Text:
        {content}
        
        Please provide the analysis in JSON format with the following structure:
        {{
            "main_licenses": ["license1", "license2"],  # List of main licenses using standard SPDX identifiers
            "is_dual_licensed": true/false,
            "dual_license_relationship": "AND/OR/none",
            "has_third_party_licenses": true/false,
            "third_party_license_location": "section name or URL where third-party licenses can be found",
            "license_relationship": "AND/OR/none",  # Relationship between main and third-party licenses
            "confidence": 0.0-1.0
        }}
        
        Note: Always use standard SPDX identifiers for licenses. Common examples:
        - Apache-2.0
        - MIT
        - GPL-2.0-or-later
        - GPL-3.0-only
        - LGPL-2.1-or-later
        - LGPL-3.0-only
        - BSD-2-Clause
        - BSD-3-Clause
        - ISC
        - MPL-2.0
        - AGPL-3.0
        """
    }
}

# Validate Gemini API configuration only if LLM is enabled
if USE_LLM:
    if not GEMINI_CONFIG["api_key"]:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Initialize Gemini client
    genai.configure(api_key=GEMINI_CONFIG["api_key"])
    logger.info(f"Initialized Gemini API with model: {GEMINI_CONFIG['model']}")

# Enum for URL types
class Kind(Enum):
    REPO = "REPO"  # Repository URL
    DIR = "DIR"    # Directory URL
    FILE = "FILE"  # File URL

class GitHubAPI:
    """GitHub API client for repository analysis.
    
    This class handles all interactions with the GitHub API, including:
    - Repository information retrieval
    - License information fetching
    - Tree structure analysis
    - Rate limit handling
    """
    
    BASE_URL = "https://api.github.com"  # Base URL for GitHub API
    
    def __init__(self):
        """Initialize the GitHub API client with authentication and proxy settings."""
        # Get GitHub token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.error("GITHUB_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN environment variable not set")
        
        logger.info("Initializing GitHub API client")
        
        # Set up headers for API requests
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
        """Make a request to GitHub API with rate limit handling.
        
        Args:
            endpoint: API endpoint to call
            params: Optional query parameters
            
        Returns:
            Dict containing the API response
            
        Raises:
            requests.exceptions.HTTPError: If the API request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Making request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        while True:
            response = self.session.get(url, params=params)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                # Handle rate limiting
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            logger.debug(f"Request successful. Status code: {response.status_code}")
            return response.json()

    def get_repo_info(self, owner: str, repo: str) -> Dict:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Dict containing repository information
        """
        logger.info(f"Fetching repository info for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}")

    def get_branches(self, owner: str, repo: str) -> List[Dict]:
        """Get all branches for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of branch information dictionaries
        """
        logger.info(f"Fetching branches for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}/branches")

    def get_tags(self, owner: str, repo: str) -> List[Dict]:
        """Get all tags for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of tag information dictionaries
        """
        logger.info(f"Fetching tags for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}/tags")

    def get_tree(self, owner: str, repo: str, sha: str) -> Dict:
        """Get repository tree structure.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA or branch name
            
        Returns:
            Dict containing the repository tree structure
        """
        logger.info(f"Fetching tree for {owner}/{repo} at {sha}")
        return self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})

    def get_license(self, owner: str, repo: str, ref: Optional[str] = None) -> Optional[Dict]:
        """Get repository license information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            ref: Optional reference (branch/tag/commit)
            
        Returns:
            Dict containing license information or None if no license found
        """
        logger.info(f"Fetching license info for {owner}/{repo} at ref: {ref}")
        try:
            params = {"ref": ref} if ref else None
            return self._make_request(f"/repos/{owner}/{repo}/license", params=params)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No license found for {owner}/{repo}")
                return None
            logger.error(f"Error fetching license: {str(e)}")
            raise

def parse_github_url(url: str) -> Tuple[str, str, Kind]:
    """Parse GitHub URL into repo URL, subpath, and kind."""
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

def resolve_version(api: GitHubAPI, owner: str, repo: str, version: Optional[str]) -> Tuple[str, bool]:
    """Resolve version to a specific ref. Returns (resolved_version, used_default_branch)."""
    logger.info(f"Resolving version for {owner}/{repo}, requested version: {version}")
    
    # Get repository info first to get the default branch
    repo_info = api.get_repo_info(owner, repo)
    default_branch = repo_info["default_branch"]
    logger.info(f"Repository default branch: {default_branch}")
    
    if not version:
        logger.debug("No version specified, using default branch")
        return default_branch, True
    
    # Convert version to string if it's a number
    version_str = str(version) if version is not None else None
    logger.debug(f"Converted version to string: {version_str}")
    
    # Extract version from package name if present (e.g. "@package@version")
    if version_str and "@" in version_str:
        version_str = version_str.split("@")[-1]
        logger.debug(f"Extracted version from package name: {version_str}")
    
    # Get all branches and tags
    branches = api.get_branches(owner, repo)
    tags = api.get_tags(owner, repo)
    
    logger.debug(f"Found {len(branches)} branches and {len(tags)} tags")
    
    # Prepare candidates for fuzzy matching
    candidates = {
        branch["name"]: branch["name"]  # Store branch name instead of SHA
        for branch in branches
    }
    candidates.update({
        tag["name"]: tag["name"]  # Store tag name instead of SHA
        for tag in tags
    })
    
    if not candidates:
        logger.warning("No branches or tags found, using default branch")
        return default_branch, True
    
    logger.debug(f"Available candidates: {list(candidates.keys())}")
    
    # Try exact match first
    if version_str in candidates:
        logger.info(f"Found exact version match: {version_str}")
        return version_str, False
    
    # Try to find a candidate that contains the version string
    for candidate in candidates:
        if version_str in candidate:
            logger.info(f"Found candidate containing version: {candidate}")
            return candidate, False
    
    # If no direct match, try fuzzy matching with different strategies
    # First try token sort ratio for better handling of version numbers
    best_match = process.extractOne(
        version_str,
        candidates.keys(),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=SCORE_THRESHOLD  # Increased minimum score threshold
    )
    
    if not best_match:
        # If token sort ratio fails, try partial ratio for better substring matching
        best_match = process.extractOne(
            version_str,
            candidates.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff= SCORE_THRESHOLD  # Increased minimum score threshold
        )
    
    if not best_match:
        logger.warning(f"No matching version found for {version_str}, using default branch")
        return default_branch, True
    
    # process.extractOne returns (matched_string, score, index)
    matched_version, score, _ = best_match
    logger.info(f"Best match: {matched_version} (score: {score})")
    
    # If score is too low, use default branch
    if score < SCORE_THRESHOLD:  # Increased minimum score threshold
        logger.warning(f"Best match score {score} is too low, using default branch")
        return default_branch, True
    
    # Check if the requested version falls within any version range
    def is_version_in_range(version_str: str, range_str: str) -> bool:
        """Check if a version falls within a version range."""
        if not range_str.endswith('.x'):
            return False
            
        range_base = range_str[:-2]  # Remove '.x'
        return version_str.startswith(range_base)
    
    # Check if the version falls within any version range
    for candidate in candidates:
        if is_version_in_range(version_str, candidate):
            logger.info(f"Version {version_str} falls within range {candidate}")
            return candidate, False
    
    # Compare versions to find the latest available version
    def extract_version_numbers(version_str: str) -> List[int]:
        """Extract numeric parts from version string."""
        # Remove any non-numeric characters except dots
        version_parts = re.findall(r'\d+', version_str)
        return [int(part) for part in version_parts]
    
    try:
        requested_version = extract_version_numbers(version_str)
        latest_version = None
        latest_version_nums = [0]  # Initialize with a very low version
        
        # Find the latest version from candidates
        for candidate in candidates:
            if candidate.endswith('.x'):
                # For version ranges, use the base version
                base_version = candidate[:-2]
                candidate_nums = extract_version_numbers(base_version)
            else:
                candidate_nums = extract_version_numbers(candidate)
            
            if candidate_nums > latest_version_nums:
                latest_version = candidate
                latest_version_nums = candidate_nums
        
        # Compare requested version with latest available version
        if requested_version > latest_version_nums:
            logger.info(f"Requested version {version_str} is newer than latest available version {latest_version}, using default branch")
            return default_branch, True
        else:
            logger.info(f"Using matched version {matched_version} as it's not newer than latest available version {latest_version}")
            return matched_version, False
            
    except Exception as e:
        logger.warning(f"Failed to compare versions: {str(e)}, using matched version")
        return matched_version, False

def find_license_files(path_map: Dict[str, Any], sub_path: str, keywords: List[str]) -> List[str]:
    """Find license files in the path map."""
    logger.info(f"Searching for license files in {sub_path or 'root'} with keywords: {keywords}")
    url_logger.info(f"Starting license file search in {sub_path or 'root'}")
    url_logger.info(f"Search keywords: {keywords}")
    results = []
    base_path = sub_path.rstrip("/")
    
    # Ensure path_map is a dictionary and has a tree key
    if not isinstance(path_map, dict) or "tree" not in path_map:
        logger.warning(f"Invalid path map format: {path_map}")
        url_logger.error(f"Invalid path map format: {path_map}")
        return results
    
    # Ensure tree is a list
    tree_items = path_map.get("tree", [])
    if not isinstance(tree_items, list):
        logger.warning(f"Tree is not a list: {tree_items}")
        url_logger.error(f"Tree is not a list: {tree_items}")
        return results
    
    # Get the resolved version from the path_map
    resolved_version = path_map.get("resolved_version", "main")
    logger.info(f"Using resolved version for URL construction: {resolved_version}")
    url_logger.info(f"Using resolved version: {resolved_version}")
    
    url_logger.info(f"Processing {len(tree_items)} tree items")
    
    # Log all paths for debugging
    url_logger.info("All paths in tree:")
    for item in tree_items:
        if isinstance(item, dict):
            path = item.get("path", "")
            type_ = item.get("type", "")
            url_logger.info(f"Path: {path}, Type: {type_}")
    
    for item in tree_items:
        # Ensure item is a dictionary
        if not isinstance(item, dict):
            logger.warning(f"Invalid tree item format: {item}")
            url_logger.error(f"Invalid tree item format: {item}")
            continue
            
        if item.get("type") != "blob":
            continue
            
        path = item.get("path", "")
        if not path:
            continue
            
        if base_path and not path.startswith(base_path):
            url_logger.debug(f"Skipping path {path} - not in base path {base_path}")
            continue
            
        name = path.lower().split("/")[-1]
        url_logger.debug(f"Checking file: {name}")
        
        if any(keyword in name for keyword in keywords):
            logger.debug(f"Found license file: {path}")
            url_logger.info(f"Found license file: {path}")
            url_logger.info(f"Matching keyword found in: {name}")
            
            # Convert GitHub API URL to GitHub web interface URL
            api_url = item.get("url", "")
            if api_url:
                try:
                    url_logger.info("Processing API URL: " + api_url)
                    
                    # Get the repository info from the API URL
                    if "/repos/" not in api_url:
                        url_logger.error(f"Invalid API URL format (no /repos/): {api_url}")
                        continue
                        
                    repo_parts = api_url.split("/repos/")[1].split("/")
                    if len(repo_parts) < 2:
                        url_logger.error(f"Invalid API URL format (insufficient parts): {api_url}")
                        continue
                        
                    owner = repo_parts[0]
                    repo = repo_parts[1]
                    
                    # Get just the filename from the path
                    filename = path.split("/")[-1]
                    
                    # Log URL construction components
                    url_logger.info("URL Construction Components:")
                    url_logger.info(f"Original path: {path}")
                    url_logger.info(f"API URL: {api_url}")
                    url_logger.info(f"Owner: {owner}")
                    url_logger.info(f"Repo: {repo}")
                    url_logger.info(f"Resolved version: {resolved_version}")
                    url_logger.info(f"Filename: {filename}")
                    
                    # Construct the GitHub web interface URL
                    web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{filename}"
                    url_logger.info(f"Constructed URL: {web_url}")
                    
                    results.append(web_url)
                    logger.debug(f"Converted API URL to web URL: {web_url}")
                except Exception as e:
                    error_msg = f"Failed to convert API URL to web URL: {str(e)}"
                    logger.warning(error_msg)
                    url_logger.error(error_msg)
                    url_logger.error(f"API URL that caused error: {api_url}")
                    results.append(api_url)  # Fallback to original URL if conversion fails
    
    url_logger.info(f"Found {len(results)} license files")
    url_logger.info(f"Final results: {results}")
    logger.info(f"Found {len(results)} license files")
    return results

def draw_file_tree(tree_items: List[Dict], indent: str = "", is_last: bool = True, prefix: str = "") -> List[str]:
    """Draw a tree structure for the repository files."""
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
                subtree = draw_file_tree(children, new_indent, is_last_item, new_prefix)
                lines.extend(subtree)
    
    return lines

def save_tree_to_file(repo_url: str, version: str, tree_items: List[Dict], log_file: str = "repository_trees.log"):
    """Save the repository tree structure to a log file."""
    logger.info(f"Saving tree structure for {repo_url} at version {version}")
    
    # Create the tree structure
    tree_lines = draw_file_tree(tree_items)
    
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
    """Get the content of a file from GitHub."""
    try:
        # Convert GitHub web URL to raw content URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Failed to get content for {path}: {str(e)}")
        return None

def find_readme(tree_items: List[Dict], sub_path: str = "") -> Optional[str]:
    """Find README file in the given path."""
    readme_patterns = ["readme", "read.me"]
    base_path = sub_path.rstrip("/")
    
    for item in tree_items:
        if item.get("type") != "blob":
            continue
            
        path = item.get("path", "")
        if not path:
            continue
            
        if base_path and not path.startswith(base_path):
            continue
            
        name = path.lower().split("/")[-1]
        if any(pattern in name for pattern in readme_patterns):
            return path
    
    return None

def analyze_license_content(content: str) -> Dict:
    """Analyze license content using Gemini API.
    
    This function uses the Gemini API to analyze license content and extract:
    - Main license types
    - Dual license relationships
    - Third-party license information
    - Copyright notices
    - License conflicts
    
    Args:
        content: The license content to analyze
        
    Returns:
        Dict containing the analysis results with the following structure:
        {
            "licenses": List[str],  # List of detected license types
            "is_dual_licensed": bool,  # Whether multiple licenses are present
            "dual_license_relationship": str,  # "AND", "OR", or "none"
            "has_third_party_licenses": bool,  # Whether third-party licenses are mentioned
            "third_party_license_location": Optional[str],  # URL or location of third-party licenses
            "copyright_notices": List[str],  # List of copyright notices
            "license_conflicts": List[str],  # List of detected license conflicts
            "analysis_summary": str  # Summary of the analysis
        }
    """
    if not USE_LLM:
        logger.info("LLM analysis disabled, returning default values")
        return {
            "licenses": [],
            "is_dual_licensed": False,
            "dual_license_relationship": "none",
            "has_third_party_licenses": False,
            "third_party_license_location": None,
            "copyright_notices": [],
            "license_conflicts": [],
            "analysis_summary": "LLM analysis disabled"
        }
    
    try:
        # Construct prompt for license analysis
        prompt = f"""Analyze the following license content and provide a detailed analysis in JSON format.
        Focus on:
        1. Main license types (e.g., MIT, Apache-2.0, GPL-3.0)
        2. Dual license relationships (AND/OR)
        3. Third-party license information
        4. Copyright notices
        5. License conflicts
        
        License content:
        {content}
        
        Provide the analysis in this JSON format:
        {{
            "main_licenses": ["license1", "license2"],
            "is_dual_licensed": true/false,
            "dual_license_relationship": "AND/OR/none",
            "has_third_party_licenses": true/false,
            "third_party_license_location": "url or location",
            "copyright_notices": ["notice1", "notice2"],
            "license_conflicts": ["conflict1", "conflict2"],
            "analysis_summary": "brief summary"
        }}
        
        Only return the JSON object, no additional text."""
        
        # Generate content using Gemini API
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        
        # Parse the response text to extract JSON
        try:
            # Find JSON content in the response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Convert main_licenses to licenses for consistency
                if "main_licenses" in result:
                    result["licenses"] = result.pop("main_licenses")
                return result
            else:
                logger.warning("No JSON found in response")
                return {
                    "licenses": [],
                    "is_dual_licensed": False,
                    "dual_license_relationship": "none",
                    "has_third_party_licenses": False,
                    "third_party_license_location": None,
                    "copyright_notices": [],
                    "license_conflicts": [],
                    "analysis_summary": "Failed to parse response"
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {
                "licenses": [],
                "is_dual_licensed": False,
                "dual_license_relationship": "none",
                "has_third_party_licenses": False,
                "third_party_license_location": None,
                "copyright_notices": [],
                "license_conflicts": [],
                "analysis_summary": f"JSON parsing error: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error in license analysis: {str(e)}")
        return {
            "licenses": [],
            "is_dual_licensed": False,
            "dual_license_relationship": "none",
            "has_third_party_licenses": False,
            "third_party_license_location": None,
            "copyright_notices": [],
            "license_conflicts": [],
            "analysis_summary": f"Analysis error: {str(e)}"
        }

def get_last_update_time(api: GitHubAPI, owner: str, repo: str, ref: str) -> str:
    """Get the last update time for a specific ref."""
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

def extract_copyright_info(content: str) -> Optional[str]:
    """Extract copyright information from text content."""
    if not USE_LLM:
        return None
        
    try:
        prompt = """
        Analyze the following text and extract copyright information.
        Look for phrases like "Copyright (c)", "Copyright ©", or similar copyright notices.
        If found, return the exact copyright notice.
        If not found, return null.
        
        Text:
        {content}
        
        Return the result in JSON format:
        {{
            "copyright_notice": "exact copyright notice if found, otherwise null"
        }}
        """
        
        llm_logger.info("Copyright Extraction Request:")
        llm_logger.info(f"Prompt: {prompt.format(content=content)}")
        
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt.format(content=content))
        
        llm_logger.info("Copyright Extraction Response:")
        llm_logger.info(f"Response: {response.text}")
        
        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                copyright_notice = result.get("copyright_notice")
                llm_logger.info(f"Extracted copyright notice: {copyright_notice}")
                return copyright_notice
            else:
                llm_logger.warning("No JSON found in copyright extraction response")
    except Exception as e:
        llm_logger.error(f"Failed to extract copyright info: {str(e)}", exc_info=True)
    return None

def construct_copyright_notice(api: GitHubAPI, owner: str, repo: str, ref: str, component_name: str, readme_content: Optional[str] = None, license_content: Optional[str] = None) -> str:
    """Construct a copyright notice if one is not found."""
    # Try to extract copyright from content first
    copyright_notice = None
    
    # Combine README and license content for analysis
    combined_content = ""
    if readme_content:
        combined_content += readme_content + "\n\n"
    if license_content:
        combined_content += license_content
    
    if combined_content:
        # Use LLM to analyze copyright information
        try:
            prompt = f"""Analyze the following text and extract ONLY the copyright information. Return ONLY the copyright notice if found, or 'None' if no copyright notice is found. Do not include any other information or explanation.

Text to analyze:
{combined_content}"""
            
            llm_logger.info("Copyright Notice Construction Request:")
            llm_logger.info(f"Prompt: {prompt}")
            
            model = genai.GenerativeModel(GEMINI_CONFIG["model"])
            response = model.generate_content(prompt)
            
            llm_logger.info("Copyright Notice Construction Response:")
            llm_logger.info(f"Response: {response.text}")
            
            if response.text:
                text = response.text.strip()
                if text and text.lower() != "none":
                    copyright_notice = text
                    llm_logger.info(f"Found copyright notice via LLM: {copyright_notice}")
                else:
                    llm_logger.info("No copyright notice found in LLM response")
        except Exception as e:
            llm_logger.error(f"Error using LLM for copyright analysis: {str(e)}", exc_info=True)
    
    # If no copyright found via LLM, construct one
    if not copyright_notice:
        year = get_last_update_time(api, owner, repo, ref)
        copyright_notice = f"Copyright (c) {year} {component_name} original author and authors"
        llm_logger.info(f"Constructed default copyright notice: {copyright_notice}")
    
    
    return copyright_notice

def find_github_url_from_package_url(package_url: str) -> Optional[str]:
    """Use LLM to find GitHub URL from a package URL (e.g., Maven, NPM)."""
    if not USE_LLM:
        logger.info("LLM is disabled, skipping GitHub URL lookup")
        return None
        
    try:
        prompt = f"""
        Given the following package URL, find the corresponding GitHub repository URL if it exists.
        Package URL: {package_url}
        
        Return the result in JSON format:
        {{
            "github_url": "https://github.com/owner/repo if found, otherwise null",
            "confidence": 0.0-1.0
        }}
        
        Only return a GitHub URL if you are confident it is the correct repository.
        If you are not sure, return null.
        """
        
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

def process_repository(url: str, github_api: GitHubAPI) -> Dict:
    """Process a single repository to analyze its license information.
    
    This function performs a comprehensive analysis of a repository's license information,
    including:
    1. URL validation and parsing
    2. Repository information retrieval
    3. License file detection and analysis
    4. Copyright notice extraction
    5. License conflict detection
    
    Args:
        url: The GitHub repository URL to analyze
        github_api: Initialized GitHubAPI instance
        
    Returns:
        Dict containing the analysis results with the following structure:
        {
            "input_url": str,  # Original input URL
            "resolved_url": str,  # Resolved repository URL
            "owner": str,  # Repository owner
            "repo": str,  # Repository name
            "license_type": str,  # Primary license type
            "license_file_path": str,  # Path to license file
            "license_content": str,  # Content of license file
            "license_analysis": Dict,  # Detailed license analysis
            "copyright_notices": List[str],  # List of copyright notices
            "license_conflicts": List[str],  # List of license conflicts
            "error": Optional[str]  # Error message if any
        }
    """
    # Initialize result dictionary with default values
    result = {
        "input_url": url,
        "resolved_url": None,
        "owner": None,
        "repo": None,
        "license_type": None,
        "license_file_path": None,
        "license_content": None,
        "license_analysis": None,
        "copyright_notices": [],
        "license_conflicts": [],
        "error": None
    }
    
    try:
        # Step 1: Validate and parse URL
        logger.info(f"Processing repository: {url}")
        parsed_url = urlparse(url)
        if not parsed_url.netloc or not parsed_url.path:
            raise ValueError("Invalid GitHub URL")
            
        # Extract owner and repo from URL path
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
            
        owner, repo = path_parts[0], path_parts[1]
        result["owner"] = owner
        result["repo"] = repo
        
        # Step 2: Get repository information
        logger.info(f"Fetching repository info for {owner}/{repo}")
        repo_info = github_api.get_repo_info(owner, repo)
        result["resolved_url"] = repo_info["html_url"]
        
        # Step 3: Get license information
        logger.info("Fetching license information")
        license_info = github_api.get_license(owner, repo)
        if license_info:
            result["license_type"] = license_info["license"]["key"]
            result["license_file_path"] = license_info["path"]
            result["license_content"] = license_info["content"]
            
            # Step 4: Analyze license content
            logger.info("Analyzing license content")
            license_analysis = analyze_license_content(license_info["content"])
            result["license_analysis"] = license_analysis
            
            # Extract copyright notices and license conflicts
            if "copyright_notices" in license_analysis:
                result["copyright_notices"] = license_analysis["copyright_notices"]
            if "license_conflicts" in license_analysis:
                result["license_conflicts"] = license_analysis["license_conflicts"]
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing repository {url}: {str(e)}")
        result["error"] = str(e)
        return result

def main():
    """Main execution function for the GitHub Repository License Analyzer.
    
    This function:
    1. Loads and validates environment variables
    2. Reads input URLs from Excel file
    3. Processes each repository
    4. Generates analysis report
    5. Handles errors and logging
    """
    try:
        # Log version information
        logger.info("="*50)
        logger.info("GitHub Repository License Analyzer")
        logger.info("Version: 1.0.0")
        logger.info("="*50)
        
        # Check environment variables
        logger.info("Checking environment variables...")
        if not os.getenv("GITHUB_TOKEN"):
            raise ValueError("GITHUB_TOKEN environment variable not set")
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable not set")
        logger.info("Environment variables validated successfully")
        
        # Initialize GitHub API client
        logger.info("Initializing GitHub API client...")
        github_api = GitHubAPI()
        logger.info("GitHub API client initialized successfully")
        
        # Read input file
        logger.info("Reading input file...")
        try:
            df = pd.read_excel("input.xlsx")
            if "url" not in df.columns:
                raise ValueError("Input file must contain a 'url' column")
            logger.info(f"Successfully read {len(df)} URLs from input file")
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            raise
        
        # Process repositories
        logger.info("Starting repository processing...")
        results = []
        for index, row in df.iterrows():
            url = row["url"]
            logger.info(f"Processing repository {index + 1}/{len(df)}: {url}")
            
            # Process repository
            result = process_repository(url, github_api)
            results.append(result)
            
            # Log progress
            if result["error"]:
                logger.warning(f"Error processing {url}: {result['error']}")
            else:
                logger.info(f"Successfully processed {url}")
        
        # Create results DataFrame
        logger.info("Creating results DataFrame...")
        results_df = pd.DataFrame(results)
        
        # Ensure all required columns exist
        required_columns = [
            "input_url", "resolved_url", "owner", "repo",
            "license_type", "license_file_path", "license_content",
            "is_dual_licensed", "dual_license_relationship",
            "has_third_party_licenses", "third_party_license_location",
            "copyright_notices", "license_conflicts", "error"
        ]
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in results_df.columns:
                if col in ["is_dual_licensed", "has_third_party_licenses"]:
                    results_df[col] = False
                elif col == "dual_license_relationship":
                    results_df[col] = "none"
                else:
                    results_df[col] = None
        
        # Reorder columns
        results_df = results_df[required_columns]
        
        # Save results
        logger.info("Saving results to Excel file...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"license_analysis_{timestamp}.xlsx"
        results_df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Log summary
        logger.info("="*50)
        logger.info("Analysis Summary:")
        logger.info(f"Total repositories processed: {len(results)}")
        logger.info(f"Successful analyses: {len([r for r in results if not r['error']])}")
        logger.info(f"Failed analyses: {len([r for r in results if r['error']])}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
