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
import time
import logging
from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import unquote

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from tqdm import tqdm
import google.generativeai as genai

#=============================================================================
# Import internal packages
#=============================================================================
from core.logging_utils import setup_logging
from core.github_utils import GitHubAPI
from core.config import GEMINI_CONFIG, SCORE_THRESHOLD
from core.utils import get_concluded_license

# ============================================================================
# Load Prompts Section
import yaml

from core.github_utils import process_github_repository
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# ============================================================================  
# Configuration and Setup Section
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)


# Set up logging
setup_logging()

load_dotenv()

# Configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
logger = logging.getLogger('main')  # Use the main logger for general application logging
substep_logger =logging.getLogger('substep')
url_logger = logging.getLogger('url_construction')

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
            result = process_github_repository(
                api,
                row["github_url"],
                row.get("version"),
                name=row.get("name", None)
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
            "component_name","concluded_license", "license_files","copyright_notice", "license_analysis", "license_type",
            "has_license_conflict", "readme_license", "license_file_license",
             "status", "license_determination_reason",
            "is_dual_licensed", "dual_license_relationship", "has_third_party_licenses",
            "third_party_license_location"
        ]
        output_df["concluded_license"] = output_df.apply(get_concluded_license, axis=1)
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

