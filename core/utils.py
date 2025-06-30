from typing import Any, Dict, List, Optional
from .config import GEMINI_CONFIG
import google.generativeai as genai
from dotenv import load_dotenv
import yaml
import os
import logging
import re
import json






load_dotenv()
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)
logger = logging.getLogger('main')
def is_sha_version(version: str) -> bool:
    """
    判断字符串是否为Git提交SHA（7~40位十六进制，无点号）。
    避免误判如'1.2.3'、'202406'等常规版本号。
    """
    if not isinstance(version, str):
        return False
    v = version.strip()
    # 只允许全十六进制且长度7~40且不包含点号
    return (
        7 <= len(v) <= 40 and
        '.' not in v and
        v.lower() == v and  # git sha 通常为小写
        all(c in '0123456789abcdef' for c in v)
    )


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
        llm_logger = logging.getLogger('llm_interaction')
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
    url_logger = logging.getLogger('url_construction')
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
        llm_logger = logging.getLogger('llm_interaction')

        llm_logger.info("Copyright Extraction Request:")
        llm_logger.info(f"Prompt: {prompt}")

        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)

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


def construct_copyright_notice(year: str, owner: str, repo: str, ref: str, component_name: str, readme_content: Optional[str] = None, license_content: Optional[str] = None) -> str:
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
            llm_logger = logging.getLogger('llm_interaction')

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
        copyright_notice = f"Copyright (c) {year} {component_name} original author and authors"
        llm_logger.info(f"Constructed default copyright notice: {copyright_notice}")

    return copyright_notice


