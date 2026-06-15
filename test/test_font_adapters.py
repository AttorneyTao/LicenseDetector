"""字体适配器集成测试 + 报告生成。

运行方式：
    python test/test_font_adapters.py            # 默认每类约 6 个，合计约 30
    python test/test_font_adapters.py 8          # 每类最多 8 个
    python test/test_font_adapters.py 8 10       # 每类最多 8 个，总量上限 10（先到先得）

从真实 input.xlsx 按「站点类别」做分层抽样（github / google_fonts / fontshare /
maoken / crawl_llm），长尾 crawl_llm 优先覆盖不同域名以扩大测试面。抽样确定性
（固定随机种子），样本与报告输出均已加入 .gitignore。

注意：本测试会真实访问网络（抓取页面 / 调 GitHub API），对 maoken 与长尾站点
还会调用小片段 LLM，请留意 token 成本与耗时。
"""

import os
import sys
import json
import random
import asyncio
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

# 让脚本可从项目根目录导入 core 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from core.font_utils import process_font_entry, classify_font_source  # noqa: E402

INPUT_XLSX = os.path.join(ROOT, "input.xlsx")
FIXTURE_DIR = os.path.join(ROOT, "test", "fixtures")
REPORT_DIR = os.path.join(ROOT, "test", "font_report")

SEED = 42
DEFAULT_PER_CAT = 6           # 每类抽样上限（5 类 → 约 30 个）
CATEGORY_ORDER = ["github", "google_fonts", "fontshare", "maoken", "crawl_llm"]
CONCURRENCY = 5


def build_samples(per_cat: int = DEFAULT_PER_CAT, total_cap=None) -> pd.DataFrame:
    """从 input.xlsx 分层抽样：每类最多 per_cat 个；crawl_llm 优先覆盖不同域名。"""
    df = pd.read_excel(INPUT_XLSX)
    df = df.dropna(subset=["github_url"]).copy()
    df["category"] = df["github_url"].apply(classify_font_source)
    df["domain"] = df["github_url"].apply(lambda u: urlparse(str(u)).netloc.lower())

    rng = random.Random(SEED)
    picked_idx = []

    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        if cat == "crawl_llm":
            # 按域名出现频次降序，优先覆盖高频长尾站点（微信公众号、站酷等），每个域名取一个
            dom_counts = sub["domain"].value_counts()
            chosen = []
            for dom in dom_counts.index:  # 频次从高到低
                rows = sub[sub["domain"] == dom].index.tolist()
                chosen.append(rng.choice(rows))
                if len(chosen) >= per_cat:
                    break
        else:
            idxs = sub.index.tolist()
            rng.shuffle(idxs)
            chosen = idxs[:per_cat]
        picked_idx.extend(chosen)

    if total_cap is not None:
        picked_idx = picked_idx[:total_cap]

    samples = df.loc[picked_idx, ["name", "version", "github_url", "category", "domain"]].copy()
    return samples.reset_index(drop=True)


def _write_samples_fixture(samples: pd.DataFrame) -> str:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, "font_samples.xlsx")
    samples.to_excel(path, index=False)
    return path


async def _run(per_cat: int, total_cap=None):
    from core.github_utils import GitHubAPI
    api = GitHubAPI()
    try:
        await api.initialize()
    except Exception as e:
        print(f"[WARN] GitHub API 初始化失败（github/google_fonts 样本可能受影响）: {e}")

    samples = build_samples(per_cat=per_cat, total_cap=total_cap)
    _write_samples_fixture(samples)
    print(f"抽样 {len(samples)} 个用例，类别分布: {samples['category'].value_counts().to_dict()}\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    rows = [None] * len(samples)

    async def worker(i, r):
        async with sem:
            url = r["github_url"]
            category = r["category"]
            try:
                result = await process_font_entry(api, r.get("name"), r.get("version"), url)
            except Exception as e:
                result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            analysis = result.get("license_analysis") or {}
            rows[i] = {
                "category": category,
                "input_name": r.get("name"),
                "input_url": url,
                "status": result.get("status"),
                "license_type": result.get("license_type"),
                "copyright_notice": result.get("copyright_notice"),
                "license_files": result.get("license_files"),
                "repo_url": result.get("repo_url"),
                "reason": analysis.get("license_determination_reason") if isinstance(analysis, dict) else None,
                "source": analysis.get("license_source") if isinstance(analysis, dict) else None,
                "error": result.get("error"),
            }
            print(f"[{category}] {r.get('name')} -> {result.get('status')} | {result.get('license_type')}")

    await asyncio.gather(*(worker(i, r) for i, r in samples.iterrows()))
    return rows


def _write_report(rows):
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_df = pd.DataFrame(rows)

    # Excel
    xlsx_path = os.path.join(REPORT_DIR, f"font_test_report_{ts}.xlsx")
    report_df.to_excel(xlsx_path, index=False)
    report_df.to_excel(os.path.join(REPORT_DIR, "font_test_report_latest.xlsx"), index=False)

    # 按类别汇总
    summary = (
        report_df.assign(ok=report_df["status"] == "success")
        .groupby("category")
        .agg(total=("status", "size"), success=("ok", "sum"))
        .reset_index()
    )

    ok = int((report_df["status"] == "success").sum())
    lines = [
        "# 字体适配器测试报告",
        "",
        f"- 生成时间：{ts}",
        f"- 样本数：{len(rows)}",
        f"- 成功（status=success）：{ok}/{len(rows)}",
        "",
        "## 分类别成功率",
        "",
        "| 站点类别 | 样本数 | 成功 |",
        "|---|---|---|",
    ]
    for _, s in summary.iterrows():
        lines.append(f"| {s['category']} | {int(s['total'])} | {int(s['success'])} |")

    lines += [
        "",
        "## 明细",
        "",
        "| 站点类别 | 字体名 | 状态 | 授权类型 | 版权/作者 | 授权证据URL | 判定依据 | 来源 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def c(v):
        return str(v).replace("\n", " ").replace("|", "/") if v is not None else ""

    for r in rows:
        lines.append(
            f"| {c(r['category'])} | {c(r['input_name'])} | {c(r['status'])} | "
            f"{c(r['license_type'])} | {c(r['copyright_notice'])} | {c(r['license_files'])} | "
            f"{c(r['reason'])} | {c(r['source'])} |"
        )
    lines += ["", "## 原始结果 (JSON)", "", "```json", json.dumps(rows, ensure_ascii=False, indent=2), "```"]
    md = "\n".join(lines)
    md_path = os.path.join(REPORT_DIR, f"font_test_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(REPORT_DIR, "font_test_report_latest.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n报告已生成：\n  {md_path}\n  {xlsx_path}")
    print("\n" + md.split("## 明细")[0])


def main():
    per_cat = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PER_CAT
    total_cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = asyncio.run(_run(per_cat=per_cat, total_cap=total_cap))
    _write_report(rows)


if __name__ == "__main__":
    main()
