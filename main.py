import os
import re
import json
import time
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from tqdm import tqdm
import google.generativeai as genai

SCORE_THRESHOLD = 65

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_license_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure URL construction logging
url_logger = logging.getLogger('url_construction')
url_logger.setLevel(logging.INFO)
url_handler = logging.FileHandler('url_construction.log', encoding='utf-8')
url_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
url_logger.addHandler(url_handler)

# Configure LLM logging
llm_logger = logging.getLogger('llm_interaction')
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler('llm_interaction.log', encoding='utf-8')
llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(llm_handler)

# Add substep logging
substep_logger = logging.getLogger('substep')
substep_logger.setLevel(logging.INFO)
substep_handler = logging.FileHandler('substep.log', encoding='utf-8')
substep_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
substep_logger.addHandler(substep_handler)

# Set the default encoding for stdout and stderr to utf-8
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Load environment variables
load_dotenv()

# Configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
logger.info(f"LLM analysis is {'enabled' if USE_LLM else 'disabled'}")

# Gemini API Configuration
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "model": "gemini-2.5-flash-preview-05-20",
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

class Kind(Enum):
    REPO = "REPO"
    DIR = "DIR"
    FILE = "FILE"

class GitHubAPI:
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
        """Make a request to GitHub API with rate limit handling."""
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Making request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        while True:
            response = self.session.get(url, params=params)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            logger.debug(f"Request successful. Status code: {response.status_code}")
            return response.json()

    def get_repo_info(self, owner: str, repo: str) -> Dict:
        """Get repository information."""
        logger.info(f"Fetching repository info for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}")

    def get_branches(self, owner: str, repo: str) -> List[Dict]:
        """Get all branches."""
        logger.info(f"Fetching branches for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}/branches")

    def get_tags(self, owner: str, repo: str) -> List[Dict]:
        """Get all tags."""
        logger.info(f"Fetching tags for {owner}/{repo}")
        return self._make_request(f"/repos/{owner}/{repo}/tags")

    def get_tree(self, owner: str, repo: str, sha: str) -> Dict:
        """Get repository tree."""
        logger.info(f"Fetching tree for {owner}/{repo} at {sha}")
        return self._make_request(f"/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": "1"})

    def get_license(self, owner: str, repo: str, ref: Optional[str] = None) -> Optional[Dict]:
        """Get repository license information."""
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

def analyze_license_content(content: str) -> Dict[str, Any]:
    """Use Gemini API to analyze license content."""
    if not USE_LLM:
        logger.info("LLM analysis is disabled, returning empty analysis")
        return {
            "licenses": [],
            "is_dual_licensed": False,
            "dual_license_relationship": "none",
            "license_relationship": "none",
            "confidence": 0.0,
            "third_party_license_location": None
        }
        
    try:
        prompt = GEMINI_CONFIG["prompts"]["license_analysis"].format(content=content)
        
        # Generate content using the official client
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        
        # Parse the response text into a dictionary
        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Convert main_licenses to licenses for consistency
                if "main_licenses" in result:
                    result["licenses"] = result.pop("main_licenses")
                return result
            else:
                logger.warning("No JSON found in license analysis response")
                return {
                    "licenses": [],
                    "is_dual_licensed": False,
                    "dual_license_relationship": "none",
                    "license_relationship": "none",
                    "confidence": 0.0,
                    "third_party_license_location": None
                }
    except Exception as e:
        logger.error(f"Failed to analyze license content: {str(e)}", exc_info=True)
        return {
            "licenses": [],
            "is_dual_licensed": False,
            "dual_license_relationship": "none",
            "license_relationship": "none",
            "confidence": 0.0,
            "third_party_license_location": None
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

def process_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"]
) -> Dict[str, Any]:
    """Process a single repository and return license information."""
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
        repo_info = api.get_repo_info(owner, repo)
        component_name = repo_info.get("name", repo)
        substep_logger.info(f"Retrieved component name: {component_name}")
        
        # Step 4: Resolve version
        substep_logger.info("Step 4/15: Resolving version")
        resolved_version, used_default_branch = resolve_version(api, owner, repo, version)
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
                    license_file_analysis = analyze_license_content(license_content)
                    
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
                        "copyright_notice": construct_copyright_notice(api, owner, repo, resolved_version, component_name, None, license_content),
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
        save_tree_to_file(repo_url, resolved_version, tree)
        
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
                readme_license_analysis = analyze_license_content(readme_content)
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
        copyright_notice = construct_copyright_notice(
            api, owner, repo, resolved_version, component_name,
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
                        license_file_analysis = analyze_license_content(license_content)
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
                        license_file_analysis = analyze_license_content(license_content)
                
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
                            license_file_analysis = analyze_license_content(license_content)
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
        return {
            "input_url": input_url,
            "repo_url": repo_url,
            "input_version": version,
            "resolved_version": resolved_version,
            "used_default_branch": used_default_branch,
            "component_name": component_name,
            "license_files": "",
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

def main():
    logger.info("Starting GitHub License Analyzer")
    
    # Check environment variables
    logger.info("Checking environment variables")
    github_token = os.getenv("GITHUB_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    logger.info(f"GITHUB_TOKEN present: {'Yes' if github_token else 'No'}")
    logger.info(f"GEMINI_API_KEY present: {'Yes' if gemini_api_key else 'No'}")
    
    # Initialize GitHub API
    try:
        logger.info("Step 1: Initializing GitHub API client")
        api = GitHubAPI()
        logger.info("GitHub API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GitHub API client: {str(e)}", exc_info=True)
        raise
    
    # Read input Excel file
    try:
        logger.info("Step 2: Reading input Excel file")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        df = pd.read_excel("input.xlsx")
        logger.info(f"Read {len(df)} rows from input file")
        logger.info(f"Columns found: {df.columns.tolist()}")
        logger.info(f"First row data: {df.iloc[0].to_dict()}")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}", exc_info=True)
        raise
    
    # Process each repository
    results = []
    logger.info("Step 3: Starting repository processing")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing repositories"):
        logger.info(f"Starting processing of row {idx + 1}/{len(df)}")
        logger.info(f"Processing URL: {row['github_url']}")
        logger.info(f"Version: {row.get('version')}")
        
        try:
            result = process_repository(
                api,
                row["github_url"],
                row.get("version")
            )
            
            # Extract additional license analysis fields
            if result.get("license_analysis"):
                result["is_dual_licensed"] = result["license_analysis"].get("is_dual_licensed", False)
                result["dual_license_relationship"] = result["license_analysis"].get("dual_license_relationship", "none")
                result["has_third_party_licenses"] = result["license_analysis"].get("has_third_party_licenses", False)
                result["third_party_license_location"] = result["license_analysis"].get("third_party_license_location", None)
            
            results.append(result)
            logger.info(f"Completed processing row {idx + 1}")
            
            # Save intermediate results
            try:
                pd.DataFrame(results).to_csv("temp_results.csv", index=False)
                logger.debug("Saved intermediate results")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing row {idx + 1}: {str(e)}", exc_info=True)
            results.append({
                "input_url": row["github_url"],
                "error": str(e),
                "status": "error"
            })
    
    # Create final output
    try:
        output_df = pd.DataFrame(results)
        
        # Ensure all required columns are present
        required_columns = [
            "input_url", "repo_url", "input_version", "resolved_version", "used_default_branch",
            "component_name", "license_files", "license_analysis", "license_type",
            "has_license_conflict", "readme_license", "license_file_license",
            "copyright_notice", "status", "license_determination_reason",
            "is_dual_licensed", "dual_license_relationship", "has_third_party_licenses",
            "third_party_license_location"
        ]
        
        # Add any missing columns with None values
        for col in required_columns:
            if col not in output_df.columns:
                output_df[col] = None
        
        # Reorder columns to ensure consistent output
        output_df = output_df[required_columns]
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"output_{timestamp}.xlsx"
        output_df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final results: {str(e)}", exc_info=True)
        raise
    
    # Clean up temporary file
    try:
        if os.path.exists("temp_results.csv"):
            os.remove("temp_results.csv")
            logger.debug("Removed temporary results file")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()

