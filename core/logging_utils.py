"""Application logging with daily/size rotation and seven-day retention."""

import logging
import os
import re
from datetime import date, datetime, timedelta
from logging.handlers import BaseRotatingHandler
from pathlib import Path


LOG_MAX_BYTES = 50 * 1024 * 1024
LOG_RETENTION_DAYS = 7
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class WeeklyRotatingFileHandler(BaseRotatingHandler):
    """Rotate at midnight or at the size limit, whichever happens first."""

    def __init__(self, filename, max_bytes=LOG_MAX_BYTES, retention_days=LOG_RETENTION_DAYS):
        if max_bytes <= 0 or retention_days <= 0:
            raise ValueError("Log size and retention must be positive")
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._active_date = (
            datetime.fromtimestamp(path.stat().st_mtime).date()
            if path.exists() and path.stat().st_size
            else date.today()
        )
        super().__init__(str(path), mode="a", encoding="utf-8", delay=False)
        self.prune_archives()

    def shouldRollover(self, record):
        # Uvicorn's logging reconfiguration can close an already-attached
        # handler. FileHandler normally reopens on emit, but rotating handlers
        # call shouldRollover first.
        if self.stream is None:
            self.stream = self._open()
        if date.today() != self._active_date:
            return True
        current_size = os.fstat(self.stream.fileno()).st_size
        message_size = len((self.format(record) + self.terminator).encode("utf-8"))
        return current_size > 0 and current_size + message_size > self.max_bytes

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if os.path.exists(self.baseFilename) and os.path.getsize(self.baseFilename):
            prefix = f"{self.baseFilename}.{self._active_date.isoformat()}."
            sequence = 1
            while os.path.exists(f"{prefix}{sequence:03d}"):
                sequence += 1
            os.rename(self.baseFilename, f"{prefix}{sequence:03d}")
        self._active_date = date.today()
        self.stream = self._open()
        self.prune_archives()

    def prune_archives(self):
        base = Path(self.baseFilename)
        archive_pattern = re.compile(rf"^{re.escape(base.name)}\.(\d{{4}}-\d{{2}}-\d{{2}})\.\d{{3,}}$")
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).timestamp()
        for path in base.parent.iterdir():
            match = archive_pattern.fullmatch(path.name)
            if not match or not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()


def create_log_handler(filename):
    return WeeklyRotatingFileHandler(filename)

def _is_console_handler(handler):
    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)


def _ensure_rotating_handler(logger, filename, level):
    abspath = os.path.abspath(filename)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if os.path.abspath(handler.baseFilename) != abspath:
            continue
        if isinstance(handler, WeeklyRotatingFileHandler):
            return
        logger.removeHandler(handler)
        handler.close()
    handler = create_log_handler(filename)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(level)
    logger.addHandler(handler)


def _ensure_file_logger(name, filename, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    _ensure_rotating_handler(logger, filename, logging.NOTSET)
    return logger


def setup_logging(console_level=logging.INFO):
    """Configure root and dedicated loggers without adding duplicate handlers."""
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    _ensure_rotating_handler(root, "logs/github_license_analyzer.log", logging.DEBUG)
    if not any(_is_console_handler(handler) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console_handler)
    for handler in root.handlers:
        if _is_console_handler(handler):
            handler.setLevel(console_level)

    # Some ecosystem modules attach their own handlers before setup_logging.
    log_dir = Path("logs").resolve()
    for value in logging.Logger.manager.loggerDict.values():
        if not isinstance(value, logging.Logger):
            continue
        for handler in list(value.handlers):
            if type(handler) is not logging.FileHandler:
                continue
            path = Path(handler.baseFilename)
            if path.parent != log_dir:
                continue
            level, formatter = handler.level, handler.formatter
            value.removeHandler(handler)
            handler.close()
            replacement = create_log_handler(path)
            replacement.setLevel(level)
            replacement.setFormatter(formatter)
            value.addHandler(replacement)

    logger = logging.getLogger(__name__)

    url_logger = _ensure_file_logger('url_construction', r'logs/url_construction.log')
    llm_logger = _ensure_file_logger('llm_interaction', r'logs/llm_interaction.log')
    _ensure_file_logger('llm_cache', r'logs/llm_cache.log')
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
