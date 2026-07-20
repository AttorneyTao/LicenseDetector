# -*- coding: utf-8 -*-
"""手动端到端验证脚本（需要网络 / GITHUB_TOKEN / LLM 配置）：

验证「GitHub 无对应版本 tag 时回退到带版本注册表链接」的完整流程，
并对最终输出的 license_files URL 做 HTTP 可达性检查。

用法::

    uv run python test/verify_versioned_registry_fallback.py <batch>

批次说明见文件底部 BATCHES。注意 mvnrepository.com 有 Cloudflare 反爬，
其链接的可达性检查会报 False，但浏览器访问正常（这正是版本校验走
repo1.maven.org 的原因）。
"""
import asyncio
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)  # prompts.yaml / logs 等相对路径依赖

import pandas as pd  # noqa: E402

from core.utils import is_url_reachable  # noqa: E402


async def check_and_report(tag, result):
    lf = result.get("license_files")
    reachable = await is_url_reachable(lf) if isinstance(lf, str) and lf.startswith("http") else None
    print(f"\n=== {tag} ===")
    print(f"  status              : {result.get('status')}")
    print(f"  input_version       : {result.get('input_version')}")
    print(f"  resolved_version    : {result.get('resolved_version')}")
    print(f"  used_default_branch : {result.get('used_default_branch')}")
    print(f"  license_type        : {result.get('license_type')}")
    print(f"  license_files       : {lf}")
    print(f"  license_files 可达  : {reachable}")
    # 不变量：输入了版本号时，license_files 不应是默认分支 blob 链接
    if isinstance(lf, str) and "github.com" in lf and "/blob/" in lf:
        ref = lf.split("/blob/")[1].split("/")[0]
        verdict = "OK(带tag/SHA)" if ref not in ("master", "main") else "!! 默认分支链接"
        print(f"  blob ref            : {ref}  -> {verdict}")


async def run_repos(rows):
    from main import process_all_repos
    from core.github_utils import GitHubAPI

    api = GitHubAPI()
    df = pd.DataFrame(rows)
    results = await process_all_repos(api, df, max_concurrency=2)
    for row, result in zip(rows, results):
        await check_and_report(row["github_url"] + " @ " + str(row.get("version")), result)


async def run_pypi(cases):
    from core.pypi_utils import process_pypi_repository

    for url, version in cases:
        result = await process_pypi_repository(url, version)
        await check_and_report(f"{url} @ {version}", result)


BATCHES = {
    # Maven + Go 对照组：怪异 tag 格式(v_1.7.36) / 正常 tag(v1.9.0) / monorepo 子路径 tag(storage/v1.30.1)
    # 期望：全部命中 tag，保留 GitHub 带 tag blob 链接
    "1": lambda: run_repos([
        {"github_url": "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "version": "1.7.36", "name": "slf4j-api"},
        {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": "1.9.0", "name": "atomic"},
        {"github_url": "https://pkg.go.dev/cloud.google.com/go/storage", "version": "1.30.1", "name": "storage"},
    ]),
    # npm / crates.io 回归对照：行为应保持不变（tag 命中 → blob 链接）
    "2": lambda: run_repos([
        {"github_url": "https://www.npmjs.com/package/left-pad", "version": "1.3.0", "name": "left-pad"},
        {"github_url": "https://crates.io/crates/serde", "version": "1.0.190", "name": "serde"},
    ]),
    # PyPI 对照组：requests 有精确 tag；psycopg2 有同名 tag
    "3": lambda: run_pypi([
        ("https://pypi.org/project/requests/", "2.31.0"),
        ("https://pypi.org/project/psycopg2/", "2.9.9"),
    ]),
    # 真实"无 tag"用例：
    #  - jsr305: 仓库 amaembo/jsr-305 零 tag（LLM 大概率找不到仓库 → 走既有 Maven POM 回退，同样输出带版本链接）
    #  - atomic@9.9.9: 仓库无 tag 且 proxy 也无此版本 → 期望保持默认分支链接（负路径：拼接链接无效不替换）
    "4": lambda: run_repos([
        {"github_url": "https://mvnrepository.com/artifact/com.google.code.findbugs/jsr305", "version": "3.0.2", "name": "jsr305"},
        {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": "9.9.9", "name": "atomic"},
    ]),
    # Maven 正向触发新门控：listenablefuture 的占位版本号在 guava 仓库必无 tag，
    # 但 Maven Central 真实存在该版本 → 期望替换为 mvnrepository 带版本链接
    "5": lambda: run_repos([
        {"github_url": "https://mvnrepository.com/artifact/com.google.guava/listenablefuture",
         "version": "9999.0-empty-to-avoid-conflict-with-guava", "name": "listenablefuture"},
    ]),
    # PyPI 正向触发新门控：types-requests 指向 python/typeshed（零 tag 仓库），
    # PyPI 真实存在 2.31.0.6 → 期望替换为 pypi.org 带版本链接
    "6": lambda: run_pypi([
        ("https://pypi.org/project/types-requests/", "2.31.0.6"),
    ]),
}

if __name__ == "__main__":
    batch = sys.argv[1] if len(sys.argv) > 1 else "1"
    asyncio.run(BATCHES[batch]())
