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

def setup_logging():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
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

    # Configure npm logging
    npm_logger = logging.getLogger('npm')
    npm_logger.setLevel(logging.INFO)
    npm_handler = logging.FileHandler(r'logs/npm.log', encoding='utf-8')
    npm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    npm_logger.addHandler(npm_handler)

    # Configure maven logging
    maven_logger = logging.getLogger('maven_utils')
    maven_logger.setLevel(logging.INFO)
    maven_handler = logging.FileHandler(r'logs/maven.log', encoding='utf-8')
    maven_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    maven_logger.addHandler(maven_handler)


    return {
        "main": logger,
        "url": url_logger,
        "llm": llm_logger,
        "substep": substep_logger,
        "version_resolve": version_resolve_logger,
        "npm": npm_logger,
        "maven": maven_logger
    }