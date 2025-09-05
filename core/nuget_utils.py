import aiohttp
import os
import logging
import platform
from openai import AsyncOpenAI
from core.config import QWEN_CONFIG
import yaml
from bs4 import BeautifulSoup
import json
import asyncio
import re

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "nuget_utils.log")

logger = logging.getLogger("nuget_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 加载 prompts.yaml
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

async def get_github_file_content(final_url: str) -> str:
    """
    如果 final_url 是 GitHub 文件，调用 GitHub API 获取原始内容
    """
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", final_url)
    if not match:
        return None
    owner, repo, branch, path = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.text()
    return None

async def get_license_type_by_llm(license_url: str, name: str = "", version: str = "") -> str:
    """
    获取 licenseUrl 的最终跳转地址和内容，并调用大模型判断其许可证类型
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(license_url, allow_redirects=True) as resp:
            final_url = str(resp.url)
            html = await resp.text()

    # 判断是否为 GitHub 文件
    github_file_content = await get_github_file_content(final_url)
    if github_file_content:
        content_for_llm = github_file_content
    else:
        # 只保留 body 内容，去除 head、script、style 等
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["head", "script", "style"]):
            tag.decompose()
        content_for_llm = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

    # 使用 prompts.yaml 中的 license_analysis 模板
    prompt = PROMPTS["license_analysis"].format(
        content=content_for_llm[:5000],  # 内容太长时只取前5000字符
    )
    prompt = (
        f"Package name: {name}\n"
        f"Package version: {version}\n"
        f"License URL: {final_url}\n"
        f"{prompt}"
        "reply only in json, nothing else, for those license info has complicated license rule, you need to analyse according to the package name and version"
    )
    logger.info(f"LLM 处理 license_url: {license_url}, 最终 URL: {final_url}, prompt 长度: {len(prompt)}")
    logger.info(f"LLM prompts:{prompt}")


    client = AsyncOpenAI(
        api_key=QWEN_CONFIG["api_key"],
        base_url=QWEN_CONFIG["base_url"],
    )
    if platform.system() == "Windows":

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=QWEN_CONFIG["model"],
        )
        logger.info(f"LLM Response:{response}")
        content = response.choices[0].message.content if response.choices else ""
        # 解析 JSON 返回
        try:
            result = json.loads(content)
            main_licenses = result.get("main_licenses")
            if main_licenses and isinstance(main_licenses, list) and main_licenses:
                return main_licenses[0]
        except Exception:
            logger.warning(f"LLM返回不是JSON格式: {content}")
        license_type = content.strip().splitlines()[0] if content else ""
        return license_type
    except Exception as e:
        logger.error(f"LLM 判断 license_type 失败: {e}", exc_info=True)
        return ""

async def process_nuget_packages(name: str, version: str) -> dict:
    """
    根据名称和版本查找 NuGet 包元数据，返回兼容 process_github_repository 的格式
    license_files 字段用 NuGet 包主页（如 https://www.nuget.org/packages/{name}/{version}）
    license_type 优先选择 licenseExpression
    """
    input_url = f"https://www.nuget.org/packages/{name}"
    url = f"https://api.nuget.org/v3/registration5-semver1/{name.lower()}/index.json"
    logger.info(f"处理 NuGet 包: {name}, version: {version}, url: {url}")
    github_url = None
    resolved_version = None
    license_type = None
    license_files = ""
    license_analysis = None
    readme_license = None
    license_file_license = None
    copyright_notice = None
    component_name = name

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                logger.info(f"NuGet API 响应状态: {resp.status}")
                if resp.status == 200:
                    raw_text = await resp.text()
                    logger.info(f"NuGet API 原始返回数据: {raw_text}")
                    data = await resp.json()
                    items = data.get("items", [])
                    found = False
                    entry = {}
                    # 查找指定版本
                    for page in items:
                        for pkg in page.get("items", []):
                            ver = pkg.get("catalogEntry", {}).get("version")
                            if ver and ver.lower() == version.lower():
                                entry = pkg.get("catalogEntry", {})
                                resolved_version = ver
                                # 优先选择 licenseExpression
                                license_type = entry.get("licenseExpression")
                                if not license_type and entry.get("licenseUrl"):
                                    license_type = await get_license_type_by_llm(entry.get("licenseUrl"), name, resolved_version or version)
                                github_url = entry.get("projectUrl")
                                found = True
                                break
                        if found:
                            break
                    # 如果没有找到指定版本，取最新版本
                    if not found:
                        for page in reversed(items):
                            if page.get("items"):
                                entry = page["items"][-1].get("catalogEntry", {})
                                resolved_version = entry.get("version")
                                license_type = entry.get("licenseExpression") or entry.get("licenseUrl")
                                github_url = entry.get("projectUrl")
                                break
                    # license_files 用 NuGet 包主页
                    if resolved_version:
                        license_files = f"https://www.nuget.org/packages/{name}/{resolved_version}"
    except Exception as e:
        logger.error(f"NuGet API 调用异常: {e}", exc_info=True)

    result = {
        "input_url": input_url,
        "repo_url": github_url,
        "input_version": version,
        "resolved_version": resolved_version,
        "used_default_branch": False,
        "component_name": component_name,
        "license_files": license_files,
        "license_analysis": license_analysis,
        "license_type": license_type,
        "has_license_conflict": False,
        "readme_license": readme_license,
        "license_file_license": license_file_license,
        "copyright_notice": copyright_notice,
        "status": "success" if license_type else "skipped",
        "license_determination_reason": "Found by NuGet API" if license_type else "Not found or no license info in NuGet API"
    }
    logger.info(f"NuGet 包处理结果: {result}")
    return result

async def check_if_nuget_package_exists(name: str, version: str) -> bool:
    """
    只要根据名称能够匹配 NuGet 包，就返回 True
    """
    url = f"https://api.nuget.org/v3/registration5-semver1/{name.lower()}/index.json"
    logger.info(f"检查 NuGet 包是否存在: {name}, version: {version}, url: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                logger.info(f"NuGet API 响应状态: {resp.status}")
                if resp.status == 200:
                    raw_text = await resp.text()
                    logger.info(f"NuGet API 原始返回数据: {raw_text}")
                    data = await resp.json()
                    if "items" in data and len(data["items"]) > 0:
                        logger.info(f"NuGet 包 {name} 存在")
                        return True
    except Exception as e:
        logger.error(f"NuGet API 调用异常: {e}", exc_info=True)
    logger.info(f"NuGet 包 {name} 不存在")
    return False