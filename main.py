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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_license_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        self.session = requests.Session()
        self.session.headers.update(self.headers)

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
        score_cutoff=60
    )
    
    if not best_match:
        # If token sort ratio fails, try partial ratio for better substring matching
        best_match = process.extractOne(
            version_str,
            candidates.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=60
        )
    
    if not best_match:
        logger.warning(f"No matching version found for {version_str}, using default branch")
        return default_branch, True
    
    # process.extractOne returns (matched_string, score, index)
    matched_version, score, _ = best_match
    logger.info(f"Best match: {matched_version} (score: {score})")
    
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
    results = []
    base_path = sub_path.rstrip("/")
    
    # Ensure path_map is a dictionary and has a tree key
    if not isinstance(path_map, dict) or "tree" not in path_map:
        logger.warning(f"Invalid path map format: {path_map}")
        return results
    
    # Ensure tree is a list
    tree_items = path_map.get("tree", [])
    if not isinstance(tree_items, list):
        logger.warning(f"Tree is not a list: {tree_items}")
        return results
    
    # Get the resolved version from the path_map
    resolved_version = path_map.get("resolved_version", "main")  # Changed from "sha" to "resolved_version"
    logger.info(f"Using resolved version for URL construction: {resolved_version}")
    
    for item in tree_items:
        # Ensure item is a dictionary
        if not isinstance(item, dict):
            logger.warning(f"Invalid tree item format: {item}")
            continue
            
        if item.get("type") != "blob":
            continue
            
        path = item.get("path", "")
        if not path:
            continue
            
        if base_path and not path.startswith(base_path):
            continue
            
        name = path.lower().split("/")[-1]
        if any(keyword in name for keyword in keywords):
            logger.debug(f"Found license file: {path}")
            # Convert GitHub API URL to GitHub web interface URL
            api_url = item.get("url", "")
            if api_url:
                # Extract owner, repo, and path from the API URL
                # Example API URL: https://api.github.com/repos/owner/repo/git/blobs/01c82a159046ea6337d3fe1bc771c03b874d166b
                # Convert to: https://github.com/owner/repo/blob/branch/path
                try:
                    # Get the repository info from the API URL
                    repo_parts = api_url.split("/repos/")[1].split("/")
                    owner = repo_parts[0]
                    repo = repo_parts[1]
                    
                    # Construct the GitHub web interface URL using the resolved version
                    web_url = f"https://github.com/{owner}/{repo}/blob/{resolved_version}/{path}"
                    results.append(web_url)
                    logger.debug(f"Converted API URL to web URL: {web_url}")
                except Exception as e:
                    logger.warning(f"Failed to convert API URL to web URL: {str(e)}")
                    results.append(api_url)  # Fallback to original URL if conversion fails
    
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
        
        logger.info("Making request to Gemini API:")
        logger.info(f"Model: {GEMINI_CONFIG['model']}")
        logger.info(f"Prompt: {prompt}")
        
        # Generate content using the official client
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        
        logger.info(f"Raw API Response: {response}")
        
        if response.text:
            logger.info(f"Extracted text response: {response.text}")
            
            # Extract JSON from the response text
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"Parsed license analysis result: {json.dumps(analysis, indent=2)}")
                
                # Convert the new format to the expected format
                licenses = analysis["main_licenses"]
                if analysis.get("has_third_party_licenses"):
                    licenses.append("third-party")
                
                return {
                    "licenses": licenses,
                    "is_dual_licensed": analysis["is_dual_licensed"],
                    "dual_license_relationship": analysis.get("dual_license_relationship", "none"),
                    "license_relationship": analysis["license_relationship"],
                    "confidence": analysis["confidence"],
                    "third_party_license_location": analysis.get("third_party_license_location")
                }
            else:
                logger.warning("No JSON found in the response text")
        else:
            logger.warning("No text in the API response")
        
        logger.warning("No valid analysis found in Gemini API response")
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
        
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt.format(content=content))
        
        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("copyright_notice")
    except Exception as e:
        logger.warning(f"Failed to extract copyright info: {str(e)}")
    return None

def construct_copyright_notice(api: GitHubAPI, owner: str, repo: str, ref: str, component_name: str, readme_content: Optional[str] = None, license_content: Optional[str] = None) -> str:
    """Construct a copyright notice if one is not found."""
    # Try to extract copyright from content first
    if readme_content:
        copyright_notice = extract_copyright_info(readme_content)
        if copyright_notice:
            return copyright_notice
            
    if license_content:
        copyright_notice = extract_copyright_info(license_content)
        if copyright_notice:
            return copyright_notice
    
    # If no copyright found, construct one
    year = get_last_update_time(api, owner, repo, ref)
    return f"Copyright (c) {year} {component_name} original author and authors"

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
        
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        
        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                github_url = result.get("github_url")
                confidence = result.get("confidence", 0.0)
                
                if github_url and confidence >= 0.7:  # Only accept if confidence is high enough
                    logger.info(f"Found GitHub URL: {github_url} with confidence {confidence}")
                    return github_url
                else:
                    logger.info(f"No confident GitHub URL match found (confidence: {confidence})")
    except Exception as e:
        logger.warning(f"Failed to find GitHub URL: {str(e)}")
    return None

def process_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"]
) -> Dict[str, Any]:
    """Process a single repository and return license information."""
    logger.info(f"Processing repository: {github_url} (version: {version})")
    try:
        # Store original input URL
        input_url = github_url
        
        # Check if the URL is a GitHub URL
        parsed = urlparse(github_url)
        if parsed.netloc != "github.com":
            logger.info(f"Non-GitHub URL detected: {github_url}")
            github_url = find_github_url_from_package_url(github_url)
            if not github_url:
                logger.warning(f"Could not find GitHub URL for {github_url}")
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
            logger.info(f"Found corresponding GitHub URL: {github_url}")
        
        # Parse URL
        repo_url, sub_path, kind = parse_github_url(github_url)
        owner, repo = repo_url.split("/")[-2:]
        logger.debug(f"Parsed URL: owner={owner}, repo={repo}, kind={kind}, sub_path={sub_path}")
        
        # Get repository info for component name
        repo_info = api.get_repo_info(owner, repo)
        component_name = repo_info.get("name", repo)
        logger.info(f"Retrieved component name: {component_name}")
        
        # Resolve version
        resolved_version, used_default_branch = resolve_version(api, owner, repo, version)
        logger.info(f"Resolved version to: {resolved_version}, used_default_branch: {used_default_branch}")
        
        # Get tree using the resolved version
        tree_response = api.get_tree(owner, repo, resolved_version)
        if not isinstance(tree_response, dict):
            logger.error(f"Invalid tree response: {tree_response}")
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
            logger.error(f"Invalid tree data: {tree}")
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
            
        logger.debug(f"Retrieved tree with {len(tree)} items")
        
        # Save tree structure to file
        save_tree_to_file(repo_url, resolved_version, tree)
        
        # First try to find README in subpath
        readme_path = find_readme(tree, sub_path)
        if not readme_path:
            # If no README in subpath, try repo root
            readme_path = find_readme(tree)
        
        readme_content = None
        readme_license_analysis = None
        if readme_path:
            logger.info(f"Found README at: {readme_path}")
            readme_content = get_file_content(api, owner, repo, readme_path, resolved_version)
            if readme_content:
                readme_license_analysis = analyze_license_content(readme_content)
                if readme_license_analysis["licenses"]:
                    logger.info(f"Found license information in README: {readme_license_analysis}")
        
        # Search for license files
        path_map = {
            "tree": tree,
            "resolved_version": resolved_version
        }
        license_files = find_license_files(path_map, sub_path, license_keywords)
        logger.info(f"Found {len(license_files)} license files in subpath")
        
        # Get license content for copyright analysis
        license_content = None
        if license_files:
            for license_file in license_files:
                license_content = get_file_content(api, owner, repo, license_file.split("/")[-1], resolved_version)
                if license_content:
                    break
        
        # Get copyright notice
        copyright_notice = construct_copyright_notice(
            api, owner, repo, resolved_version, component_name,
            readme_content, license_content
        )
        logger.info(f"Copyright notice: {copyright_notice}")
        
        # If licenses found in subpath, analyze them
        if license_files:
            license_paths = [url.split("/")[-1] for url in license_files]
            determination_reason = f"Found license files in {sub_path or 'root'}: {', '.join(license_paths)}"
            logger.info(determination_reason)
            
            # If no license info found in README, analyze license files
            license_file_analysis = None
            if not readme_license_analysis or not readme_license_analysis["licenses"]:
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
                    logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")
            
            # Use license file analysis if available, otherwise use README analysis
            final_analysis = license_file_analysis or readme_license_analysis
            
            return {
                "input_url": input_url,
                "repo_url": repo_url,
                "input_version": version,
                "resolved_version": resolved_version,
                "used_default_branch": used_default_branch,
                "component_name": component_name,
                "license_files": "\n".join(license_files),
                "license_analysis": final_analysis,
                "license_type": final_analysis["licenses"][0] if final_analysis and final_analysis["licenses"] else None,
                "has_license_conflict": license_conflict,
                "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
                "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": determination_reason
            }
        
        # If no licenses found, try repo-level license
        logger.info("No license files found in subpath, trying repo-level license")
        try:
            license_info = api.get_license(owner, repo, ref=resolved_version)
            if license_info:
                license_type = license_info.get("license", {}).get("spdx_id")
                determination_reason = f"License determined via GitHub API: {license_type}"
                logger.info(determination_reason)
                
                # If no license info found in README, analyze the license content
                license_file_analysis = None
                if not readme_license_analysis or not readme_license_analysis["licenses"]:
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
                        logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")
                
                # Use license file analysis if available, otherwise use README analysis
                final_analysis = license_file_analysis or readme_license_analysis
                
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": resolved_version,
                    "used_default_branch": used_default_branch,
                    "component_name": component_name,
                    "license_type": license_type,
                    "license_files": license_info["download_url"],
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
                logger.warning(f"No license found for {owner}/{repo} at version {resolved_version}")
            else:
                logger.error(f"Error fetching license: {str(e)}")
        
        # If still no licenses, search entire repo
        if not license_files:
            logger.info("No repo-level license found, searching entire repository")
            license_files = find_license_files(path_map, "", license_keywords)
            logger.info(f"Found {len(license_files)} license files in entire repository")
            
            if license_files:
                license_paths = [url.split("/")[-1] for url in license_files]
                determination_reason = f"Found license files in repository root: {', '.join(license_paths)}"
                logger.info(determination_reason)
                
                # If no license info found in README, analyze license files
                license_file_analysis = None
                if not readme_license_analysis or not readme_license_analysis["licenses"]:
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
                        logger.warning(f"License conflict detected between README ({readme_licenses}) and license file ({file_licenses})")
                
                # Use license file analysis if available, otherwise use README analysis
                final_analysis = license_file_analysis or readme_license_analysis
                
                return {
                    "input_url": input_url,
                    "repo_url": repo_url,
                    "input_version": version,
                    "resolved_version": resolved_version,
                    "used_default_branch": used_default_branch,
                    "component_name": component_name,
                    "license_files": "\n".join(license_files),
                    "license_analysis": final_analysis,
                    "license_type": final_analysis["licenses"][0] if final_analysis and final_analysis["licenses"] else None,
                    "has_license_conflict": license_conflict,
                    "readme_license": readme_license_analysis["licenses"][0] if readme_license_analysis and readme_license_analysis["licenses"] else None,
                    "license_file_license": license_file_analysis["licenses"][0] if license_file_analysis and license_file_analysis["licenses"] else None,
                    "copyright_notice": copyright_notice,
                    "status": "success",
                    "license_determination_reason": determination_reason
                }
        
        # If no licenses found anywhere
        determination_reason = "No license files found in repository"
        logger.warning(determination_reason)
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
        logger.error(f"Error processing repository {github_url}: {error_msg}", exc_info=True)
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
    
    # Initialize GitHub API
    try:
        api = GitHubAPI()
        logger.info("GitHub API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GitHub API client: {str(e)}", exc_info=True)
        raise
    
    # Read input Excel file
    try:
        logger.info("Reading input Excel file")
        df = pd.read_excel("input.xlsx")
        logger.info(f"Read {len(df)} rows from input file")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}", exc_info=True)
        raise
    
    # Process each repository
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing repositories"):
        logger.info(f"Processing row {idx + 1}/{len(df)}")
        result = process_repository(
            api,
            row["github_url"],
            row.get("version")
        )
        results.append(result)
        
        # Save intermediate results
        try:
            pd.DataFrame(results).to_csv("temp_results.csv", index=False)
            logger.debug("Saved intermediate results")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}", exc_info=True)
    
    # Create final output
    try:
        output_df = pd.DataFrame(results)
        
        # Ensure all required columns are present
        required_columns = [
            "input_url", "repo_url", "input_version", "resolved_version", "used_default_branch",
            "component_name", "license_files", "license_analysis", "license_type",
            "has_license_conflict", "readme_license", "license_file_license",
            "copyright_notice", "status", "license_determination_reason"
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
