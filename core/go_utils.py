from typing import Optional
import aiohttp
import os
import logging

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "go_utils.log")

logger = logging.getLogger("go_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def pkggo_to_proxy_url(pkggo_url: str) -> str:
    """
    将各种 pkg.go.dev/go module 地址（含参数、锚点、无协议等）转换为 proxy.golang.org 的元数据 API 地址
    例如:
        https://pkg.go.dev/go.uber.org/atomic -> https://proxy.golang.org/go.uber.org/atomic/@latest
        pkg.go.dev/go.uber.org/atomic?tab=doc -> https://proxy.golang.org/go.uber.org/atomic/@latest
        go.uber.org/atomic -> https://proxy.golang.org/go.uber.org/atomic/@latest
    """
    import re
    # 去掉协议
    url = pkggo_url.strip()
    url = re.sub(r"^https?://", "", url)
    # 去掉域名，只保留模块路径
    match = re.match(r"(pkg\.go\.dev/|go\.dev/)?(?P<module>[a-zA-Z0-9\.\-_\/]+)", url)
    if not match:
        raise ValueError(f"无法识别 Go module 路径: {pkggo_url}")
    module_path = match.group("module")
    # 去掉参数和锚点
    module_path = module_path.split("?", 1)[0].split("#", 1)[0]
    result = f"https://proxy.golang.org/{module_path}/@latest"
    logger.info(f"转换 {pkggo_url} 为 {result}")
    return result


async def get_github_url_from_pkggo(pkggo_url: str, version: Optional[str] = None, name: Optional[str] = None) -> dict:
    """
    根据 pkg.go.dev/go module 地址，获取 proxy.golang.org 的元数据，并尝试提取 GitHub 仓库地址
    返回 dict: {"github_url": ..., "module_path": ..., "raw_info": ...}
    """
    proxy_url = pkggo_to_proxy_url(pkggo_url)
    async with aiohttp.ClientSession() as session:
        async with session.get(proxy_url) as resp:
            if resp.status != 200:
                logger.warning(f"请求 {proxy_url} 失败，状态码: {resp.status}")
                return {"github_url": None, "module_path": None, "raw_info": None}
            info = await resp.json()
            logger.info(f"API原始返回数据: {info}")
            # 优先从 Origin.URL 获取 github 地址
            github_url = None

            origin = info.get("Origin") or {}
            url = origin.get("URL", "")
            if url.startswith("https://github.com/") or url.startswith("http://github.com/"):
                github_url = url
            # 兼容 module_path 以 github.com/ 开头的情况
            module_path = url.replace("https://", "").replace("http://", "") if github_url else info.get("Path") or info.get("path") or ""
            if not github_url and module_path.startswith("github.com/"):
                github_url = f"https://{module_path}"
            return {
                "github_url": github_url,
                "module_path": module_path,
                "raw_info": info
            }