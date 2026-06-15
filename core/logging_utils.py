# ============================================================================
# Logging Configuration Section
# ============================================================================
# Sets up different loggers for various aspects of the application:
# - Main application logger: Tracks overall application flow and errors
# - URL construction logger: Specifically logs URL parsing and construction steps
# - LLM interaction logger: Records all interactions with the Gemini LLM
# - Substep execution logger: Tracks detailed progress of repository processing
# Each logger writes to both console and dedicated log files with timestamps

import logging
import os   
import sys
import codecs

def _is_console_handler(h):
    """判断是否为输出到控制台的 StreamHandler（排除 FileHandler 子类）。"""
    return isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)


def _ensure_file_logger(name, filename, level=logging.INFO):
    """为命名 logger 绑定一个文件 handler；幂等，避免重复调用导致日志重复。"""
    lg = logging.getLogger(name)
    lg.setLevel(level)
    abspath = os.path.abspath(filename)
    for h in lg.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abspath:
            return lg  # 已绑定，跳过
    handler = logging.FileHandler(filename, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    lg.addHandler(handler)
    return lg


def setup_logging(console_level=logging.INFO):
    """配置日志。

    Args:
        console_level: 控制台(StreamHandler)输出级别。文件始终记录 DEBUG 起的完整日志；
            CLI 模式可传 logging.WARNING，让控制台保持干净（只留进度条），详细日志仍写入 logs/。
    """
    os.makedirs('logs', exist_ok=True)

    root = logging.getLogger()
    if not root.handlers:
        # 首次配置：文件记录全量，控制台单独建 handler 以便单独调级别
        root.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(r'logs/github_license_analyzer.log', encoding='utf-8')
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        root.addHandler(console_handler)

    # 始终按入参刷新控制台 handler 的级别（支持二次调用调整）
    for h in root.handlers:
        if _is_console_handler(h):
            h.setLevel(console_level)

    logger = logging.getLogger(__name__)

    # 各专用日志（写入独立文件；幂等绑定）
    url_logger = _ensure_file_logger('url_construction', r'logs/url_construction.log')
    llm_logger = _ensure_file_logger('llm_interaction', r'logs/llm_interaction.log')
    substep_logger = _ensure_file_logger('substep', r'logs/substep.log')
    version_resolve_logger = _ensure_file_logger('version_resolve', r'logs/version_resolve.log')
    npm_logger = _ensure_file_logger('npm', r'logs/npm.log')
    maven_logger = _ensure_file_logger('maven_utils', r'logs/maven.log')

    return {
        "main": logger,
        "url": url_logger,
        "llm": llm_logger,
        "substep": substep_logger,
        "version_resolve": version_resolve_logger,
        "npm": npm_logger,
        "maven": maven_logger
    }