import os
import re
import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from core.utils import analyze_license_content, analyze_license_content_async
from core.utils import extract_copyright_info
from core.utils import construct_copyright_notice_async
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import datetime

logger = logging.getLogger("googlesource_util")

def parse_googlesource_url(url: str) -> Optional[Dict[str, str]]:
    """
    解析 googlesource 仓库 URL，支持多种格式，返回 host、project、ref、subpath
    """
    parsed = urlparse(url)
    if not parsed.netloc.endswith("googlesource.com"):
        return None

    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]

    project = ""
    ref = ""
    subpath = ""
    # 处理带有 +/ 的格式
    if "/+/" in path:
        project, after = path.split("/+/", 1)
        # after 可能是 refs/heads/main/README.md 或 main/LICENSE 或 commit_hash 或 commit_hash/README.md
        parts = after.split("/", 1)
        ref = parts[0]
        if len(parts) == 2:
            subpath = parts[1]
        project = project.strip("/")
    else:
        project = path.strip("/")
        ref = ""  # 默认用 main 或 master

    return {
        "host": parsed.netloc,
        "project": project,
        "ref": ref,
        "subpath": subpath
    }

def get_license_file_url(host: str, project: str, ref: str) -> Optional[str]:
    """
    递归查找 googlesource 某个 commit 下所有目录，寻找 LICENSE/COPYING 文件
    """
    license_names = [
        "LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "LICENSES", "LICENSE.rst", "NOTICE", "NOTICE.txt","NOTICE.md"
    ]
    visited = set()

    def walk_dirs(path: str) -> Optional[str]:
        # 构造目录页面URL
        url = f"https://{host}/{project}/+/{ref}/{path}"
        if not url.endswith('/'):
            url += '/'
        resp = requests.get(url)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # 查找所有文件链接
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 文件名匹配
            for lic_name in license_names:
                if href.endswith(f"/{lic_name}?format=TEXT"):
                    # 构造原始内容URL
                    raw_url = f"https://{host}{href}"
                    return raw_url
        # 查找所有子目录
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 只递归子目录（目录链接通常以 /+/ref/dir/ 结尾且不含 ?format=TEXT）
            if href.startswith(f"/{project}/+/{ref}/") and href.endswith('/') and "?format=TEXT" not in href:
                subdir = href.split(f"/+/{ref}/", 1)[1]
                if subdir not in visited:
                    visited.add(subdir)
                    found = walk_dirs(subdir)
                    if found:
                        return found
        return None

    # 从根目录递归查找
    return walk_dirs("")

def get_license_file_content(license_url: str) -> Optional[str]:
    """
    获取googlesource的LICENSE文件内容（base64解码）
    """
    resp = requests.get(license_url)
    if resp.status_code == 200:
        import base64
        try:
            return base64.b64decode(resp.text).decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"解码license内容失败: {e}")
    return None

async def get_license_file_url_async(host: str, project: str, ref: str, logger=None) -> Optional[str]:
    """
    异步查找 googlesource 某个 commit 下根目录和一级子目录，寻找 LICENSE/COPYING 文件
    """
    license_names = [
        "LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "LICENSES", "LICENSE.rst"
    ]
    visited = set()

    async def walk_dirs(path: str, session: aiohttp.ClientSession, depth: int = 0) -> Optional[str]:
        url = f"https://{host}/{project}/+/{ref}/{path}"
        if not url.endswith('/'):
            url += '/'
        if logger:
            logger.debug(f"正在访问目录页面: {url}")
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    if logger:
                        logger.debug(f"目录页面访问失败: {url} status={resp.status}")
                    return None
                text = await resp.text()
        except Exception as e:
            if logger:
                logger.warning(f"请求目录页面异常: {url} error={e}")
            return None
        soup = BeautifulSoup(text, "html.parser")
        # 查找所有文件链接
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 提取文件名
            filename = href.split("/")[-1]
            # 匹配 LICENSE/COPYING/NOTICE 开头的文件（允许任意后缀，大小写不敏感）
            if re.match(r"^(LICENSE|COPYING|NOTICE)(\.[\w\-.]+)?$", filename, re.IGNORECASE):
                # 拼接原始内容URL
                if "?" in href:
                    href = href.split("?", 1)[0]
                raw_url = f"https://{host}{href}?format=TEXT"
                if logger:
                    logger.info(f"发现LICENSE文件: {raw_url}")
                return raw_url
        # 只递归一层子目录
        if depth == 0:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith(f"/{project}/+/{ref}/") and href.endswith('/') and "?format=TEXT" not in href:
                    subdir = href.split(f"/+/{ref}/", 1)[1]
                    if subdir not in visited:
                        visited.add(subdir)
                        found = await walk_dirs(subdir, session, depth=1)
                        if found:
                            return found
        return None

    async with aiohttp.ClientSession() as session:
        return await walk_dirs("", session, depth=0)

# 修改主流程调用
async def process_googlesource_component_async(component_name: str, version: str, repo_url: str) -> Dict[str, Any]:
    logger.info(f"处理googlesource组件: {component_name}, {version}, {repo_url}")
    result = {
        "input_url": repo_url,
        "repo_url": repo_url,
        "input_version": version,
        "resolved_version": version,
        "used_default_branch": False,
        "component_name": component_name,
        "license_files": None,
        "license_analysis": None,
        "license_type": None,
        "has_license_conflict": None,
        "readme_license": None,
        "license_file_license": None,
        "copyright_notice": None,
        "status": "error",
        "license_determination_reason": None,
        "readme": None,
        "error": None
    }
    info = parse_googlesource_url(repo_url)
    if not info:
        result["error"] = "URL解析失败"
        return result

    host, project, ref, subpath = info["host"], info["project"], info["ref"], info["subpath"]
    ref_final = ref or (version if version and version.lower() != "无" else "main")

    # 1. 异步递归获取license文件链接
    license_url = await get_license_file_url_async(host, project, ref_final, logger=logger)
    if not license_url:
        result["error"] = "未找到LICENSE文件"
        return result
    result["license_files"] = license_url

    # 2. 获取license内容（同步即可，因只有一个请求）
    license_content = get_license_file_content(license_url)
    if not license_content:
        result["error"] = "无法获取LICENSE内容"
        return result

    # 3. LLM分析license类型（异步）
    license_analysis = await analyze_license_content_async(license_content)
    licenses = license_analysis.get("licenses", []) if license_analysis else []
    result["license_analysis"] = license_analysis
    result["license_type"] = licenses[0] if licenses else None
    result["license_file_license"] = licenses[0] if licenses else None

    # 4. 版权声明提取与LLM辅助
    year = str(datetime.datetime.now().year)
    # owner、repo、ref 可根据 googlesource url 解析获得，若无可传空字符串
    copyright_notice = await construct_copyright_notice_async(
        year=year,
        owner="",  # googlesource 可留空
        repo="",   # googlesource 可留空
        ref=version or "",
        component_name=component_name,
        readme_content=None,
        license_content=license_content
    )
    result["copyright_notice"] = copyright_notice

    result["status"] = "success"
    result["license_determination_reason"] = "Fetched from googlesource LICENSE"
    return result

# 批量处理CSV（异步版本）
async def process_csv_async(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = await process_googlesource_component_async(
            str(row.get("name", "")),
            str(row.get("version", "")),
            str(row.get("github_url", ""))
        )
        results.append(res)
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"已保存结果到 {output_csv}")

# 兼容主流程调用
def process_googlesource_component(component_name: str, version: str, repo_url: str) -> Dict[str, Any]:
    """同步包装，供主流程run_in_executor调用"""
    return asyncio.run(process_googlesource_component_async(component_name, version, repo_url))

def process_github_repository(api, github_url: str, version: str, license_keywords: list = None, name: str = None):
    """
    处理 GitHub 仓库，获取版本、许可证等信息
    """
    # 解析 GitHub URL
    parsed_url = urlparse(github_url)
    repo_name = os.path.basename(parsed_url.path)
    owner_repo = repo_name.split(".git")[0]  # 去掉 .git 后缀
    subpath = ""

    # 处理不同情况的 GitHub URL
    if parsed_url.netloc == "github.com":
        # 标准 GitHub URL
        owner, repo = owner_repo.split("/", 1)
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
    else:
        # 非标准 URL，尝试从 package URL 中提取 GitHub 地址
        logger.warning(f"非标准 GitHub URL，尝试从中提取: {github_url}")
        github_url_candidate = find_github_url_from_package_url(github_url)
        if not github_url_candidate:
            logger.error(f"无法从 {github_url} 中提取 GitHub URL")
            return None
        logger.info(f"提取的 GitHub URL: {github_url_candidate}")
        return process_github_repository(api, github_url_candidate, version, license_keywords, name)

    # 获取仓库信息
    repo_info = api.repos.get(owner=owner_repo.split("/")[0], repo=owner_repo.split("/")[1])
    if not repo_info:
        logger.error(f"无法获取仓库信息: {github_url}")
        return None

    # 获取默认分支
    default_branch = repo_info.default_branch or "main"

    # 处理版本
    resolved_version = version
    if version and version.lower() != "无":
        if re.match(r"^\d+\.\d+\.\d+$", version):
            # 精确版本
            resolved_version = version
        else:
            # 模糊版本，尝试解析为标签或分支
            tags = api.git.list_tags(owner=owner_repo.split("/")[0], repo=owner_repo.split("/")[1])
            branches = api.git.list_branches(owner=owner_repo.split("/")[0], repo=owner_repo.split("/")[1])
            all_versions = [tag.name for tag in tags] + [branch.name for branch in branches]
            matched_versions = [v for v in all_versions if version in v]
            resolved_version = matched_versions[0] if matched_versions else default_branch

    # 获取许可证信息
    license_files = []
    license_analysis = None
    license_type = None
    has_license_conflict = False
    readme_license = None
    license_file_license = None
    copyright_notice = None
    status = "success"
    license_determination_reason = "Fetched from GitHub repository"

    # 许可证文件关键字
    license_keywords = license_keywords or [
        "LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "LICENSES", "LICENSE.rst"
    ]

    # 查找许可证文件
    for keyword in license_keywords:
        try:
            file_content = api.repos.get_content(
                owner=owner_repo.split("/")[0],
                repo=owner_repo.split("/")[1],
                path=keyword,
                ref=resolved_version
            )
            if isinstance(file_content, list):
                # 目录情况，递归查找
                for item in file_content:
                    if item.type == "file" and item.name.lower() == keyword.lower():
                        file_content = api.repos.get_content(
                            owner=owner_repo.split("/")[0],
                            repo=owner_repo.split("/")[1],
                            path=item.path,
                            ref=resolved_version
                        )
                        break
            if isinstance(file_content, dict) and file_content.get("content"):
                # 成功获取文件内容，进行解码
                import base64
                decoded_content = base64.b64decode(file_content["content"]).decode("utf-8", errors="replace")
                license_files.append(keyword)
                # LLM分析license类型
                analysis = analyze_license_content(decoded_content)
                if analysis and isinstance(analysis, dict):
                    license_analysis = analysis
                    licenses = analysis.get("licenses", [])
                    license_type = licenses[0] if licenses else None
                    license_file_license = licenses[0] if licenses else None
                # 版权声明提取
                year = str(datetime.datetime.now().year)
                copyright_notice = construct_copyright_notice_async(
                    year=year,
                    owner="",  # GitHub 可留空
                    repo="",   # GitHub 可留空
                    ref=resolved_version,
                    component_name=name or repo_name,
                    readme_content=None,
                    license_content=decoded_content
                )
                break
        except Exception as e:
            logger.warning(f"处理许可证文件时发生错误: {e}")

    # 检查是否存在许可证冲突
    if license_analysis and isinstance(license_analysis, dict):
        detected_licenses = license_analysis.get("licenses", [])
        if len(detected_licenses) > 1:
            has_license_conflict = True
            status = "warning"
            license_determination_reason = "Detected multiple licenses"

    # 处理 README 文件中的许可证信息
    try:
        readme_content = api.repos.get_content(
            owner=owner_repo.split("/")[0],
            repo=owner_repo.split("/")[1],
            path="README.md",
            ref=resolved_version
        )
        if isinstance(readme_content, dict) and readme_content.get("content"):
            import base64
            decoded_readme = base64.b64decode(readme_content["content"]).decode("utf-8", errors="replace")
            readme_license = analyze_license_content(decoded_readme)
    except Exception as e:
        logger.warning(f"处理 README 文件时发生错误: {e}")

    return {
        "input_url": github_url,
        "repo_url": github_url,
        "input_version": version,
        "resolved_version": resolved_version,
        "used_default_branch": default_branch != resolved_version,
        "component_name": name or repo_name,
        "license_files": license_files,
        "license_analysis": license_analysis,
        "license_type": license_type,
        "has_license_conflict": has_license_conflict,
        "readme_license": readme_license,
        "license_file_license": license_file_license,
        "copyright_notice": copyright_notice,
        "status": status,
        "license_determination_reason": license_determination_reason
    }

def find_github_url_from_package_url(package_url: str) -> Optional[str]:
    """
    从包的 URL 中提取 GitHub 地址
    """
    # 简单处理，假设 GitHub 地址在 URL 中
    match = re.search(r"(https?://[^/]+/[^/]+/[^/]+)", package_url)
    return match.group(1) if match else None

# 兼容主流程调用
def process_github_component(component_name: str, version: str, repo_url: str, license_keywords: Optional[list] = None) -> Dict[str, Any]:
    """同步包装，供主流程run_in_executor调用"""
    api = None  # 初始化 API 对象
    return process_github_repository(api, repo_url, version, license_keywords, component_name)

