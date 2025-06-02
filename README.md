# GitHub Repository License Analyzer

A powerful tool built with Cursor to analyze license information from GitHub repositories. This tool helps identify and verify license information, detect license conflicts, and extract copyright notices from repositories.

## Features

- **License Analysis**: Analyzes both README files and license files to determine the project's license
- **License Conflict Detection**: Identifies conflicts between licenses mentioned in README and actual license files
- **Copyright Notice Extraction**: Extracts copyright information from repository content
- **Version Resolution**: Intelligently resolves version references to find the correct branch/tag
- **Non-GitHub URL Support**: Can find corresponding GitHub repositories from package URLs (Maven, NPM, etc.)
- **LLM Integration**: Uses Google's Gemini API for intelligent license analysis (can be disabled)

## Requirements

- Python 3.8+
- GitHub Personal Access Token
- Google Gemini API Key (optional, for LLM features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AttorneyTao/LicenseDetector.git
cd LicenseDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```env
GITHUB_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
USE_LLM=true  # Set to false to disable LLM features
```

## Usage

1. Prepare an input Excel file (`input.xlsx`) with columns:
   - `github_url`: Repository URL (GitHub, Maven, NPM, etc.)
   - `version`: Optional version reference

2. Run the analyzer:
```bash
python main.py
```

3. Check the output Excel file for results, including:
   - License type and analysis
   - License conflicts
   - Copyright notices
   - Version resolution information

## Output Fields

- `input_url`: Original input URL
- `repo_url`: GitHub repository URL
- `input_version`: Requested version
- `resolved_version`: Actual version used
- `used_default_branch`: Whether default branch was used
- `component_name`: Repository name
- `license_type`: Main license type
- `has_license_conflict`: Whether license conflicts were found
- `copyright_notice`: Extracted copyright information
- And more...

## Development

This project was built using [Cursor](https://cursor.sh), an AI-powered code editor that enhances development productivity.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
