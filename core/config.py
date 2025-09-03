import os
import yaml
from dotenv import load_dotenv
load_dotenv()
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "model": "gemini-2.5-flash-preview-05-20"
}

SCORE_THRESHOLD = 65

# 添加最大并发数配置
MAX_CONCURRENCY = 20

# 结果列的显示顺序配置
RESULT_COLUMNS_ORDER = [
    "input_url",
    "repo_url",
    "component_name",
    "input_version",
    "resolved_version",
    "used_default_branch",
    "concluded_license",
    "license_files",
    "copyright_notice",
    "license_type",    
    "license_analysis",
    "has_license_conflict",
    "readme_license",
    "license_file_license",    
    "status",
    "license_determination_reason",
    "license_text",
    "error"
]

THIRD_PARTY_KEYWORDS = [
    "thirdparty", "third_party", "third-party",
    "vendor", "external", "deps", "libraries", "libs","lib","lib-vendor"
    "modules", "submodules", "contrib", "packages", "extern", "externals"
]