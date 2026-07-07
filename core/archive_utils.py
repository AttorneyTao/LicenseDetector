# ============================================================================
# Archive Utils Module
# ============================================================================
# 直接下载链接兜底分析：当所有现有分析路径都未能获取到 license 信息，
# 且输入 URL 是一个源码包的直接下载链接（.tar.gz / .tar.xz / .tar.bz2 /
# .tgz / .zip 等）时，下载并解压该归档，然后复用 GitHub 仓库分析流程的
# 核心逻辑（license 文件搜索 -> LLM 主许可证挑选 -> LLM 内容分析 ->
# README 兜底）对解压目录进行本地分析。
#
# 下载过程通过 [DOWNLOAD_PROGRESS] 日志单独上报进度（CLI 日志行 / Web SSE），
# 分析结束后无论成功失败都会删除下载与解压产生的临时文件。

import asyncio
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

# 支持的归档后缀（按长度倒序匹配，保证 .tar.gz 先于 .gz 之类的误判）
ARCHIVE_SUFFIXES = (
    ".tar.gz", ".tar.xz", ".tar.bz2",
    ".tgz", ".txz", ".tbz2",
    ".zip", ".tar",
)

# 下载限制
MAX_ARCHIVE_SIZE_BYTES = int(os.getenv("ARCHIVE_MAX_SIZE_BYTES", str(1024 ** 3)))  # 默认 1GB
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("ARCHIVE_DOWNLOAD_TIMEOUT", "900"))  # 总超时 15 分钟
DOWNLOAD_RETRIES = 2  # 网络失败额外重试次数
DOWNLOAD_CHUNK_SIZE = 8192

# 进度上报节流：进度每前进 10% 或距上次上报超过 2 秒才输出一条
PROGRESS_STEP_PCT = 10.0
PROGRESS_MIN_INTERVAL_SECONDS = 2.0

# 读取 license/README 内容送 LLM 时的长度上限，防止异常大文件撑爆提示词
MAX_CONTENT_CHARS = 100_000

LICENSE_KEYWORDS = ["license", "licenses", "licence", "copying", "notice"]


def is_direct_archive_url(url: Any) -> bool:
    """判断 URL 是否为源码包的直接下载链接（按路径后缀识别，忽略 query 参数）。"""
    if not isinstance(url, str) or not url.strip():
        return False
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        return False
    path = urlparse(url).path.lower()
    return path.endswith(ARCHIVE_SUFFIXES)


def _archive_filename(url: str) -> str:
    """从 URL 中取出归档文件名（已解码），用于日志展示与命名猜测。"""
    return unquote(os.path.basename(urlparse(url).path)) or "archive"


def _strip_archive_suffix(filename: str) -> str:
    lower = filename.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def guess_name_version_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从下载文件名猜测组件名和版本号。
    例如 redis-7.2.7.tar.gz -> ("redis", "7.2.7")，
    dropbear-2020.81.tar.bz2 -> ("dropbear", "2020.81")，
    n8.0.1.tar.gz（GitHub archive 链接）-> (仓库名, "n8.0.1")。
    """
    stem = _strip_archive_suffix(_archive_filename(url))
    if not stem:
        return None, None

    # GitHub archive 链接：github.com/{owner}/{repo}/archive/refs/tags/{tag}.tar.gz
    # 文件名只有 tag，组件名从路径里取仓库名
    gh = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/archive/", url)
    if gh:
        return gh.group(2), stem

    # 纯版本号形式（7.1.2-12 / v1.2.3 / n8.0.1 等）没有名字可猜
    if re.match(r"^v?\d[\w.\-+]*$", stem) or re.match(r"^[a-z]\d[\w.\-+]*$", stem):
        return None, stem

    # name-1.2.3 / name_1.2.3 / name-v1.2.3 形式
    m = re.match(r"^(.*?)[-_]v?(\d[\w.\-+]*)$", stem)
    if m and m.group(1):
        return m.group(1), m.group(2)

    return stem, None


# ============================================================================
# 下载（带进度上报）
# ============================================================================

def _format_size(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f}GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.1f}MB"
    return f"{num_bytes / 1024:.1f}KB"


def _emit_progress(message: str, log_queue=None) -> None:
    """下载进度统一出口：logger（CLI/SSE 广播）+ 可选的请求级日志队列。"""
    logger.info(message)
    if log_queue is not None:
        try:
            log_queue.put_nowait(message)
        except Exception:
            pass


async def download_archive_with_progress(
    url: str,
    dest_path: str,
    log_queue=None,
) -> int:
    """
    流式下载归档到 dest_path，按节流规则输出 [DOWNLOAD_PROGRESS] 日志。
    返回下载的总字节数；下载失败抛出异常（网络错误会自动重试）。
    """
    filename = _archive_filename(url)
    last_error: Optional[Exception] = None

    for attempt in range(DOWNLOAD_RETRIES + 1):
        if attempt > 0:
            wait = 2 * attempt
            logger.warning(f"[DOWNLOAD] {filename} 下载失败，{wait}s 后重试（第 {attempt}/{DOWNLOAD_RETRIES} 次）: {last_error}")
            await asyncio.sleep(wait)
        try:
            timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS, sock_read=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    resp.raise_for_status()

                    total_bytes = resp.content_length or 0
                    if total_bytes > MAX_ARCHIVE_SIZE_BYTES:
                        raise ValueError(
                            f"归档大小 {_format_size(total_bytes)} 超过上限 "
                            f"{_format_size(MAX_ARCHIVE_SIZE_BYTES)}，放弃下载: {url}"
                        )

                    _emit_progress(
                        f"[DOWNLOAD_PROGRESS] {filename} 开始下载"
                        + (f"，总大小 {_format_size(total_bytes)}" if total_bytes else "（大小未知）"),
                        log_queue,
                    )

                    downloaded = 0
                    start_time = time.monotonic()
                    last_report_time = start_time
                    last_report_pct = 0.0

                    async with aiofiles.open(dest_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                            await f.write(chunk)
                            downloaded += len(chunk)

                            if downloaded > MAX_ARCHIVE_SIZE_BYTES:
                                raise ValueError(
                                    f"下载超过大小上限 {_format_size(MAX_ARCHIVE_SIZE_BYTES)}，中止: {url}"
                                )

                            now = time.monotonic()
                            pct = (downloaded / total_bytes * 100) if total_bytes else 0.0
                            should_report = (
                                (total_bytes and pct - last_report_pct >= PROGRESS_STEP_PCT)
                                or (now - last_report_time >= PROGRESS_MIN_INTERVAL_SECONDS)
                            )
                            if should_report:
                                elapsed = max(now - start_time, 1e-6)
                                speed = downloaded / elapsed
                                if total_bytes:
                                    msg = (
                                        f"[DOWNLOAD_PROGRESS] {filename} {pct:.1f}% "
                                        f"({_format_size(downloaded)}/{_format_size(total_bytes)}, "
                                        f"{_format_size(speed)}/s)"
                                    )
                                else:
                                    msg = (
                                        f"[DOWNLOAD_PROGRESS] {filename} 已下载 {_format_size(downloaded)} "
                                        f"({_format_size(speed)}/s)"
                                    )
                                _emit_progress(msg, log_queue)
                                last_report_time = now
                                last_report_pct = pct

                    elapsed = max(time.monotonic() - start_time, 1e-6)
                    _emit_progress(
                        f"[DOWNLOAD_PROGRESS] {filename} 下载完成，共 {_format_size(downloaded)}，"
                        f"耗时 {elapsed:.1f}s",
                        log_queue,
                    )
                    return downloaded
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            last_error = e
            continue  # 网络类错误重试
        # ValueError（超限）等直接向外抛，不重试

    raise RuntimeError(f"下载失败（已重试 {DOWNLOAD_RETRIES} 次）: {url}，最后错误: {last_error}")


# ============================================================================
# 解压
# ============================================================================

# Python 3.12+ 的 tar 提取过滤器错误（旧版本不存在该异常，用空元组占位）
_TAR_FILTER_ERRORS = getattr(tarfile, "FilterError", ())


def _safe_extract_tar(archive_path: str, dest_dir: str) -> None:
    """逐成员解压 tar，跳过路径穿越/危险链接等可疑成员而不是整体失败。"""
    dest_real = os.path.realpath(dest_dir)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(dest_dir, member.name))
            if not (target == dest_real or target.startswith(dest_real + os.sep)):
                logger.warning(f"跳过可疑的归档成员（路径穿越）: {member.name}")
                continue
            try:
                try:
                    # Python 3.12+ 的 data 过滤器额外拦截绝对路径、危险链接等
                    tar.extract(member, path=dest_dir, filter="data")
                except TypeError:
                    tar.extract(member, path=dest_dir)
            except _TAR_FILTER_ERRORS as e:
                logger.warning(f"跳过可疑的归档成员（{type(e).__name__}）: {member.name}")
            except Exception as e:
                logger.warning(f"解压归档成员失败，跳过 {member.name}: {e}")


def _safe_extract_zip(archive_path: str, dest_dir: str) -> None:
    dest_real = os.path.realpath(dest_dir)
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            target = os.path.realpath(os.path.join(dest_dir, info.filename))
            if target == dest_real or target.startswith(dest_real + os.sep):
                zf.extract(info, path=dest_dir)
            else:
                logger.warning(f"跳过可疑的归档成员（路径穿越）: {info.filename}")


def extract_archive(archive_path: str, dest_dir: str) -> str:
    """
    解压归档到 dest_dir，返回用于分析的根目录。
    若归档解压出唯一的顶层目录（如 redis-7.2.7/），返回该目录，否则返回 dest_dir。
    """
    os.makedirs(dest_dir, exist_ok=True)
    lower = archive_path.lower()
    if lower.endswith(".zip"):
        _safe_extract_zip(archive_path, dest_dir)
    else:
        _safe_extract_tar(archive_path, dest_dir)

    entries = os.listdir(dest_dir)
    if len(entries) == 1:
        only = os.path.join(dest_dir, entries[0])
        if os.path.isdir(only):
            return only
    return dest_dir


# ============================================================================
# 本地目录分析（复用 GitHub 仓库分析流程的核心逻辑）
# ============================================================================

def build_local_tree(root_dir: str) -> List[Dict[str, str]]:
    """
    把本地目录 walk 成与 GitHub tree API 相同的结构
    （[{"path": 相对路径, "type": "blob"}]），以便复用现有的
    find_license_files_detailed / find_readme 等函数。
    """
    tree: List[Dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 跳过隐藏目录（.git 等）
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root_dir)
            tree.append({"path": rel.replace(os.sep, "/"), "type": "blob", "url": ""})
    return tree


async def _read_local_file(root_dir: str, rel_path: str) -> Optional[str]:
    filepath = os.path.join(root_dir, rel_path.replace("/", os.sep))
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        return content[:MAX_CONTENT_CHARS] if content else None
    except Exception as e:
        logger.warning(f"读取解压文件失败 {rel_path}: {e}")
        return None


async def analyze_extracted_directory(
    root_dir: str,
    source_url: str,
    component_name: str,
    input_version: Optional[str] = None,
    resolved_version: Optional[str] = None,
    font_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    对解压后的目录执行 license 分析，流程仿照 process_github_repository：
    1. 在根目录（及 license/docs 等标准目录）搜索 license 文件；
       没有则全目录搜索；
    2. 多个 license 文件时用 LLM 挑选主要 license 文件；
    3. LLM 分析 license 内容得到 SPDX 表达式；
    4. 都没有则分析 README 中的许可证声明；
    5. 构造版权声明。

    找到 license 信息时返回与 GitHub 流程兼容的结果 dict，否则返回 None。
    """
    from .utils import (
        analyze_license_content_async,
        construct_copyright_notice_async,
        find_license_files_detailed,
        find_readme,
        find_top_level_thirdparty_dirs_local,
        prepare_license_text,
    )
    from .github_utils import select_primary_license_file

    tree = build_local_tree(root_dir)
    logger.info(f"[ARCHIVE] 解压目录共 {len(tree)} 个文件，开始本地 license 分析: {root_dir}")
    path_map = {"tree": tree, "resolved_version": resolved_version or ""}

    thirdparty_dirs = find_top_level_thirdparty_dirs_local(root_dir)

    # README 内容（用于 license 兜底分析和版权声明构造）
    readme_content: Optional[str] = None
    readme_path = find_readme(tree)
    if readme_path:
        readme_content = await _read_local_file(root_dir, readme_path)

    # Step 1: 根目录及标准目录搜索 license 文件
    license_files_detailed = find_license_files_detailed(path_map, "", LICENSE_KEYWORDS)
    if not license_files_detailed:
        # Step 2: 全目录搜索（对应 GitHub 流程的 Step 14）
        license_files_detailed = find_license_files_detailed(
            path_map, "", LICENSE_KEYWORDS, include_all_dirs=True
        )
    logger.info(f"[ARCHIVE] 找到 {len(license_files_detailed)} 个 license 文件")

    license_content: Optional[str] = None
    license_analysis: Optional[Dict[str, Any]] = None
    selected_rel_path: Optional[str] = None

    if license_files_detailed:
        selected = await select_primary_license_file(license_files_detailed, font_mode=font_mode)
        if selected:
            selected_rel_path = selected["path"]
            logger.info(f"[ARCHIVE] 选定主要 license 文件: {selected_rel_path}")
            license_content = await _read_local_file(root_dir, selected_rel_path)
            if license_content:
                license_analysis = await analyze_license_content_async(
                    license_content, f"{source_url} ({selected_rel_path})"
                )

    year = str(datetime.now().year)

    if license_analysis and license_analysis.get("licenses"):
        if "thirdparty_dirs" not in license_analysis:
            license_analysis["thirdparty_dirs"] = thirdparty_dirs
        license_type = license_analysis.get("spdx_expression") or license_analysis["licenses"][0]
        copyright_notice = await construct_copyright_notice_async(
            year, component_name, component_name, resolved_version or "",
            component_name, readme_content, license_content
        )
        return {
            "input_url": source_url,
            "repo_url": source_url,
            "input_version": input_version,
            "resolved_version": resolved_version or "",
            "used_default_branch": False,
            "component_name": component_name,
            "license_files": f"{source_url} ({selected_rel_path})",
            "license_analysis": license_analysis,
            "license_type": license_type,
            "has_license_conflict": False,
            "readme_license": None,
            "license_file_license": license_type,
            "copyright_notice": copyright_notice,
            "license_text": prepare_license_text(license_content),
            "status": "success",
            "license_determination_reason": (
                f"Downloaded source archive and analyzed license file: {selected_rel_path}"
            ),
        }

    # README 兜底（对应 GitHub 流程的 Step 15）
    if readme_content:
        readme_analysis = await analyze_license_content_async(
            readme_content, f"{source_url} ({readme_path})"
        )
        if readme_analysis and readme_analysis.get("licenses"):
            if "thirdparty_dirs" not in readme_analysis:
                readme_analysis["thirdparty_dirs"] = thirdparty_dirs
            license_type = readme_analysis.get("spdx_expression") or readme_analysis["licenses"][0]
            copyright_notice = await construct_copyright_notice_async(
                year, component_name, component_name, resolved_version or "",
                component_name, readme_content, None
            )
            return {
                "input_url": source_url,
                "repo_url": source_url,
                "input_version": input_version,
                "resolved_version": resolved_version or "",
                "used_default_branch": False,
                "component_name": component_name,
                "license_files": f"{source_url} ({readme_path})",
                "license_analysis": readme_analysis,
                "license_type": license_type,
                "has_license_conflict": False,
                "readme_license": license_type,
                "license_file_license": None,
                "copyright_notice": copyright_notice,
                "status": "success",
                "license_determination_reason": (
                    f"Downloaded source archive; found license info in README: {readme_analysis['licenses']}"
                ),
            }

    logger.info(f"[ARCHIVE] 解压目录中未找到可识别的 license 信息: {source_url}")
    return None


# ============================================================================
# 总入口
# ============================================================================

async def process_direct_archive_url(
    url: str,
    version: Optional[str] = None,
    name: Optional[str] = None,
    font_mode: bool = False,
    log_queue=None,
) -> Optional[Dict[str, Any]]:
    """
    直接下载链接兜底分析总入口：下载 -> 解压 -> 本地分析 -> 清理临时文件。
    找到 license 信息时返回标准结果 dict，任何环节失败或未找到时返回 None
    （调用方保留原有的失败结果）。
    """
    if not is_direct_archive_url(url):
        return None

    guessed_name, guessed_version = guess_name_version_from_url(url)
    component_name = name or guessed_name or _strip_archive_suffix(_archive_filename(url))
    resolved_version = version or guessed_version

    logger.info(f"[ARCHIVE] 开始归档兜底分析: {url} (组件: {component_name}, 版本: {resolved_version})")
    if log_queue is not None:
        try:
            log_queue.put_nowait(f"[INFO] 检测到直接下载链接，尝试下载归档进行分析: {url}")
        except Exception:
            pass

    tmp_dir = tempfile.mkdtemp(prefix="archive_dl_")
    try:
        archive_path = os.path.join(tmp_dir, _archive_filename(url))
        await download_archive_with_progress(url, archive_path, log_queue=log_queue)

        extract_dir = os.path.join(tmp_dir, "extracted")
        loop = asyncio.get_running_loop()
        # 解压可能较耗时（大包），放到线程池避免阻塞事件循环
        analysis_root = await loop.run_in_executor(None, extract_archive, archive_path, extract_dir)
        logger.info(f"[ARCHIVE] 解压完成，分析根目录: {analysis_root}")

        result = await analyze_extracted_directory(
            analysis_root,
            source_url=url,
            component_name=component_name,
            input_version=version,
            resolved_version=resolved_version,
            font_mode=font_mode,
        )
        if result:
            logger.info(f"[ARCHIVE] 归档兜底分析成功: {url} -> {result.get('license_type')}")
        return result
    except Exception as e:
        logger.warning(f"[ARCHIVE] 归档兜底分析失败 {url}: {e}")
        if log_queue is not None:
            try:
                log_queue.put_nowait(f"[WARNING] 归档下载分析失败: {e}")
            except Exception:
                pass
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"[ARCHIVE] 已清理临时文件: {tmp_dir}")
