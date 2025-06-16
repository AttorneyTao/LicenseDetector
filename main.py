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
from core.github_utils import save_github_tree_to_file
from core.github_utils import get_file_content
from core.github_utils import get_github_last_update_time
from core.logging_utils import setup_logging
from core.github_utils import GitHubAPI, find_github_url_from_package_url, resolve_github_version
from core.config import GEMINI_CONFIG, SCORE_THRESHOLD

# ============================================================================
# Load Prompts Section
import yaml

from core.utils import find_readme
from core.utils import analyze_license_content
from core.utils import find_license_files
from core.utils import construct_copyright_notice
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
setup_logging()
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(r'logs/github_license_analyzer.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configure URL construction logging
# url_logger = logging.getLogger('url_construction')
# url_logger.setLevel(logging.INFO)
# url_handler = logging.FileHandler(r'logs/url_construction.log', encoding='utf-8')
# url_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# url_logger.addHandler(url_handler)

# # Configure LLM logging
# llm_logger = logging.getLogger('llm_interaction')
# llm_logger.setLevel(logging.INFO)
# llm_handler = logging.FileHandler(r'logs/llm_interaction.log', encoding='utf-8')
# llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# llm_logger.addHandler(llm_handler)

# # Add substep logging
# substep_logger = logging.getLogger('substep')
# substep_logger.setLevel(logging.INFO)
# substep_handler = logging.FileHandler(r'logs/substep.log', encoding='utf-8')
# substep_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# substep_logger.addHandler(substep_handler)


# # Configure version resolve logging
# version_resolve_logger = logging.getLogger('version_resolve')
# version_resolve_logger.setLevel(logging.INFO)
# version_resolve_handler = logging.FileHandler(r'logs/version_resolve.log', encoding='utf-8')
# version_resolve_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# version_resolve_logger.addHandler(version_resolve_handler)


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





# ============================================================================
# License Analysis Functions
# ============================================================================
# Implements functions for:
# - Analyzing license content using LLM
# - Extracting copyright information
# - Constructing copyright notices
# - Finding GitHub URLs from package URLs
# - Handling different license formats and types

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
                        "copyright_notice": construct_copyright_notice(get_github_last_update_time(api,owner,repo,resolved_version), owner, repo, resolved_version, component_name, None, license_content),
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

