# ============================================================================
# Configuration and Setup Section
# ============================================================================
# This section handles all the initial setup including:
# - Importing required libraries for HTTP requests, JSON processing, and data analysis
# - Setting up logging configuration with multiple handlers for different aspects
# - Loading environment variables for API keys and configuration
# - Configuring API settings for GitHub and Gemini LLM
# - Setting up UTF-8 encoding for proper character handling

# Set the default encoding for stdout and stderr to utf-8
import sys
import codecs




# ============================================================================
# Import Required Libraries Section
# ============================================================================
import os
import re
import json
import time
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from tqdm import tqdm
import google.generativeai as genai

#=============================================================================
# Import internal packages
#=============================================================================
from core.github_utils import resolve_github_version
from core.github_utils import parse_github_url
from core.logging_utils import setup_logging
from core.github_utils import GitHubAPI, find_github_url_from_package_url, resolve_github_version
from core.config import GEMINI_CONFIG, SCORE_THRESHOLD

# ============================================================================
# Load Prompts Section
import yaml
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# ============================================================================  
# Configuration and Setup Section
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)



# ============================================================================
# Logging Configuration Section
# ============================================================================
# Sets up different loggers for various aspects of the application:
# - Main application logger: Tracks overall application flow and errors
# - URL construction logger: Specifically logs URL parsing and construction steps
# - LLM interaction logger: Records all interactions with the Gemini LLM
# - Substep execution logger: Tracks detailed progress of repository processing
# Each logger writes to both console and dedicated log files with timestamps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'logs/github_license_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure URL construction logging
url_logger = logging.getLogger('url_construction')
url_logger.setLevel(logging.INFO)
url_handler = logging.FileHandler(r'logs/url_construction.log', encoding='utf-8')
url_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
url_logger.addHandler(url_handler)

# Configure LLM logging
llm_logger = logging.getLogger('llm_interaction')
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler(r'logs/llm_interaction.log', encoding='utf-8')
llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(llm_handler)

# Add substep logging
substep_logger = logging.getLogger('substep')
substep_logger.setLevel(logging.INFO)
substep_handler = logging.FileHandler(r'logs/substep.log', encoding='utf-8')
substep_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
substep_logger.addHandler(substep_handler)


# Configure version resolve logging
version_resolve_logger = logging.getLogger('version_resolve')
version_resolve_logger.setLevel(logging.INFO)
version_resolve_handler = logging.FileHandler(r'logs/version_resolve.log', encoding='utf-8')
version_resolve_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
version_resolve_logger.addHandler(version_resolve_handler)


#sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
#sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# ============================================================================
# Environment and API Configuration Section
# ============================================================================
# Handles:
# - Loading environment variables from .env file
# - Configuring LLM settings including model selection and prompts
# - Setting up API keys for GitHub and Gemini
# - Configuring proxy settings if available
# - Validating API configurations before use

# Load environment variables
load_dotenv()

# Configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
logger.info(f"LLM analysis is {'enabled' if USE_LLM else 'disabled'}")


sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)



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



def find_license_files(path_map: Dict[str, Any], sub_path: str, keywords: List[str]) -> List[str]:
    """
    Finds license files in a repository tree.
    
    This function:
    - Searches for files matching license keywords
    - Handles different license file naming conventions
    - Converts API URLs to web URLs
    - Supports searching in specific subpaths
    - Logs search process and results
    
    Args:
        path_map (Dict[str, Any]): Repository tree structure
            Must contain:
            - tree: List of file/directory items
            - resolved_version: Version being analyzed
        sub_path (str): Subpath to search in
        keywords (List[str]): Keywords to match against filenames
            Common keywords: ['license', 'licenses', 'copying', 'notice']
            
    Returns:
        List[str]: List of URLs to license files
            Each URL is a GitHub web interface URL
    """
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
                subtree = draw_file_tree(children, new_indent, is_last_item, new_prefix)
                lines.extend(subtree)
    
    return lines

def save_tree_to_file(repo_url: str, version: str, tree_items: List[Dict], log_file: str = "repository_trees.log"):
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

def find_readme(tree_items: List[Dict], sub_path: str = "") -> Optional[str]:
    """
    Finds README file in the repository tree.
    
    This function:
    - Searches for common README file patterns
    - Checks both specified subpath and root
    - Handles different README naming conventions
    - Supports case-insensitive matching
    
    Args:
        tree_items (List[Dict]): Repository tree structure
        sub_path (str): Subpath to search in
            Default: "" (search in root)
            
    Returns:
        Optional[str]: Path to README file if found, None otherwise
            Example: "docs/README.md"
    """
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

# ============================================================================
# License Analysis Functions
# ============================================================================
# Implements functions for:
# - Analyzing license content using LLM
# - Extracting copyright information
# - Constructing copyright notices
# - Finding GitHub URLs from package URLs
# - Handling different license formats and types

def analyze_license_content(content: str) -> Dict[str, Any]:
    """
    Analyzes license content using the Gemini LLM.
    
    This function:
    - Uses Gemini API for license analysis
    - Detects license type and version
    - Identifies dual licensing
    - Finds third-party license references
    - Provides confidence scores
    
    Args:
        content (str): License content to analyze
            Can be:
            - Full license text
            - License file content
            - README license section
            
    Returns:
        Dict[str, Any]: Analysis results including:
            - licenses: List of detected licenses (SPDX identifiers)
            - is_dual_licensed: Whether multiple licenses are present
            - dual_license_relationship: How licenses relate (AND/OR)
            - has_third_party_licenses: Whether third-party licenses are mentioned
            - third_party_license_location: Where to find third-party licenses
            - license_relationship: How main and third-party licenses relate
            - confidence: Analysis confidence score (0.0-1.0)
    """
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
        prompt = PROMPTS["license_analysis"].format(content=content)
        
        # Generate content using the official client
        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        llm_logger.info("License Analysis Request:")
        llm_logger.info(f"Prompt: {prompt}")
        response = model.generate_content(prompt)
        llm_logger.info(f"License Analysis Response:{response.text}")
        
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

def extract_copyright_info(content: str) -> Optional[str]:
    """
    Extracts copyright information from text content.
    
    This function:
    - Uses LLM to analyze text
    - Identifies copyright notices
    - Handles different copyright formats
    - Extracts year and owner information
    
    Args:
        content (str): Text content to analyze
            Can be:
            - License file content
            - README content
            - Source file headers
            
    Returns:
        Optional[str]: Copyright notice if found, None otherwise
            Example: "Copyright (c) 2024 John Doe"
    """
    if not USE_LLM:
        return None
        
    try:
        # prompt = """
        # Analyze the following text and extract copyright information.
        # Look for phrases like "Copyright (c)", "Copyright ©", or similar copyright notices.
        # If found, return the exact copyright notice.
        # If not found, return null.
        
        # Text:
        # {content}
        
        # Return the result in JSON format:
        # {{
        #     "copyright_notice": "exact copyright notice if found, otherwise null"
        # }}
        # """
        prompt = PROMPTS["copyright_extract"].format(content=content)
        
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
    """
    Constructs a copyright notice for a component.
    
    This function:
    - Extracts existing copyright notices
    - Falls back to repository metadata
    - Uses LLM for analysis if available
    - Constructs default notice if needed
    
    Args:
        api (GitHubAPI): GitHub API client
        owner (str): Repository owner
        repo (str): Repository name
        ref (str): Reference (branch/tag/commit)
        component_name (str): Name of the component
        readme_content (Optional[str]): README content
        license_content (Optional[str]): License content
        
    Returns:
        str: Constructed copyright notice
            Example: "Copyright (c) 2024 Component Name original author and authors"
    """
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
            # prompt = f"""Analyze the following text and extract ONLY the copyright information. Return ONLY the copyright notice if found, or 'None' if no copyright notice is found. Do not include any other information or explanation.

            #             Text to analyze:
            #             {combined_content}
            #         """
            prompt = PROMPTS["copyright_analysis"].format(combined_content=combined_content)
            
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



def process_repository(
    api: GitHubAPI,
    github_url: str,
    version: Optional[str],
    license_keywords: List[str] = ["license", "licenses", "copying", "notice"]
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

# ============================================================================
# Main Execution Function
# ============================================================================
# The main entry point that:
# - Initializes the application
# - Reads input data
# - Processes repositories
# - Generates output
# - Handles errors and cleanup

def main():
    """
    Main execution function for the GitHub License Analyzer.
    
    This function implements the complete workflow:
    1. Environment and API initialization
    2. Input file processing
    3. Repository analysis
    4. Results compilation
    5. Output generation
    6. Error handling
    7. Cleanup
    
    The function expects:
    - input.xlsx: Excel file with GitHub URLs and optional versions
    - .env file: Environment variables for API keys
    
    It produces:
    - output_{timestamp}.xlsx: Detailed analysis results
    - Multiple log files for different aspects
    - Temporary files for intermediate results
    
    Error handling:
    - Validates environment variables
    - Checks API connectivity
    - Handles file I/O errors
    - Manages API rate limits
    - Provides detailed error logging
    """
    loggers = setup_logging()
    logger = loggers["main"]
    url_logger = loggers["url"]
    llm_logger = loggers["llm"]
    substep_logger = loggers["substep"]
    version_resolve_logger = loggers["version_resolve"]
    
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
            # Process each repository
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
                pd.DataFrame(results).to_csv("temp/temp_results.csv", index=False)
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
        output_file = f"outputs/output_{timestamp}.xlsx"
        output_df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final results: {str(e)}", exc_info=True)
        raise
    
    # Clean up temporary file
    try:
        if os.path.exists("temp/temp_results.csv"):
            os.remove("temp/temp_results.csv")
            logger.debug("Removed temporary results file")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    logger.info("Processing complete")

# ============================================================================
# Script Entry Point
# ============================================================================
# Standard Python script entry point that calls the main function

if __name__ == "__main__":
    main()

