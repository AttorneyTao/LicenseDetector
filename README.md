# GitHub Repository License Analyzer

This tool analyzes GitHub repositories to determine their license information, including dual licenses, third-party licenses, and copyright notices.

## Features

- Analyzes GitHub repositories for license information
- Supports version-specific analysis
- Detects dual licenses and their relationships
- Identifies third-party licenses
- Extracts copyright notices
- Handles both direct GitHub URLs and package URLs
- Uses LLM (Gemini API) for intelligent license analysis

## Prerequisites

- Python 3.8+
- GitHub API token
- Gemini API key (for LLM analysis)
- Required Python packages (see `requirements.txt`)

## Environment Variables

Create a `.env` file with the following variables:

```
GITHUB_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
USE_LLM=true  # Set to false to disable LLM analysis
HTTP_PROXY=http://127.0.0.1:7897  # Optional: Configure if using a proxy
HTTPS_PROXY=http://127.0.0.1:7897  # Optional: Configure if using a proxy
```

## Input Format

Create an `input.xlsx` file with the following columns:
- `github_url`: The GitHub repository URL or package URL
- `version`: (Optional) Specific version to analyze

## License Analysis Logic

The program follows a 15-step process to analyze repository licenses:

1. **URL Validation**
   - Checks if the URL is a GitHub URL
   - For non-GitHub URLs, attempts to find corresponding GitHub repository

2. **URL Parsing**
   - Extracts owner, repository, and subpath information
   - Determines if the URL points to a repository, directory, or file

3. **Repository Information**
   - Fetches basic repository information
   - Gets component name and default branch

4. **Version Resolution**
   - Resolves the specified version to a specific ref
   - Falls back to default branch if version not found
   - Uses fuzzy matching for version numbers

5. **GitHub API License Check**
   - Attempts to get license information directly from GitHub API
   - Analyzes license content if found

6. **Repository Tree Analysis**
   - Gets the complete repository tree structure
   - Saves tree structure for reference

7. **README Analysis**
   - Searches for and analyzes README files
   - Extracts license information from README content

8. **License File Search**
   - Searches for license files in the specified path
   - Uses keywords: "license", "licenses", "copying", "notice"

9. **License Content Analysis**
   - Analyzes content of found license files
   - Uses LLM to determine license type and relationships

10. **Copyright Notice Extraction**
    - Extracts copyright information from license files and README
    - Constructs copyright notice if not found

11. **License Conflict Detection**
    - Compares licenses found in README and license files
    - Flags conflicts if different licenses are found

12. **Repository-wide License Search**
    - If no licenses found in specified path, searches entire repository
    - Analyzes all found license files

13. **Repository-level License Check**
    - Checks for repository-level license information
    - Analyzes repository-level license content

14. **Third-party License Detection**
    - Identifies mentions of third-party licenses
    - Records locations of third-party license information

15. **Final Analysis**
    - Compiles all license information
    - Determines final license type and relationships

## Output Format

The program generates an Excel file with the following columns:

- `input_url`: Original input URL
- `repo_url`: GitHub repository URL
- `input_version`: Requested version
- `resolved_version`: Actual version analyzed
- `used_default_branch`: Whether default branch was used
- `component_name`: Repository/component name
- `license_files`: URLs of found license files
- `license_analysis`: Detailed license analysis results
- `license_type`: Main license type (SPDX identifier)
- `has_license_conflict`: Whether license conflicts were found
- `readme_license`: License found in README
- `license_file_license`: License found in license files
- `copyright_notice`: Extracted copyright notice
- `status`: Analysis status (success/error/skipped)
- `license_determination_reason`: Explanation of license determination
- `is_dual_licensed`: Whether multiple licenses are present
- `dual_license_relationship`: Relationship between dual licenses (AND/OR/none)
- `has_third_party_licenses`: Whether third-party licenses are present
- `third_party_license_location`: Location of third-party license information

## License Analysis Details

### Dual License Detection
- Analyzes license text for multiple license mentions
- Determines relationship between licenses (AND/OR)
- Common patterns:
  - "Licensed under X OR Y"
  - "Dual licensed under X and Y"
  - "Available under either X or Y"

### Third-party License Detection
- Identifies mentions of third-party components
- Locates sections or files containing third-party licenses
- Common locations:
  - LICENSE-THIRD-PARTY files
  - Third-party directories
  - Dependencies sections in README

### Copyright Notice Extraction
- Extracts copyright information from:
  - License files
  - README files
  - Project documentation
- Constructs copyright notice if not found:
  - Uses repository creation/update year
  - Includes component name
  - Adds "original author and authors"

## Error Handling

The program handles various error conditions:
- Invalid URLs
- Missing repositories
- API rate limits
- Network issues
- Proxy connection problems
- Invalid license formats

## Logging

The program generates several log files:
- `github_license_analyzer.log`: Main program log
- `url_construction.log`: URL processing details
- `llm_interaction.log`: LLM analysis details
- `substep.log`: Step-by-step processing details
- `repository_trees.log`: Repository structure information

## Usage

1. Set up environment variables in `.env`
2. Prepare input Excel file
3. Run the program:
   ```bash
   python main.py
   ```
4. Check the output Excel file and logs

## Notes

- The program uses the Gemini API for intelligent license analysis
- GitHub API rate limits apply
- Proxy settings can be configured for network access
- Large repositories may take longer to analyze
