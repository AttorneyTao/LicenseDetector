import pandas as pd
import pytest
from openpyxl import load_workbook

from api import _generate_output
from core.utils import format_concluded_license_semicolon, get_output_risk_levels


@pytest.mark.parametrize(("expression", "expected"), [
    ("MIT AND Apache-2.0", "MIT; Apache-2.0"),
    ("MIT AND Others", "MIT AND Others"),
    ("MIT AND Apache-2.0 AND Others", "MIT; Apache-2.0 AND Others"),
    ("(MIT OR GPL-3.0) AND LGPL-2.1", "(MIT OR GPL-3.0); LGPL-2.1"),
    ("MIT OR GPL-3.0 AND LGPL-2.1", "MIT OR GPL-3.0; LGPL-2.1"),
    ("MIT and others", "MIT and others"),
    ("Brand-AND-Identifier", "Brand-AND-Identifier"),
    ("(MIT)AND(GPL-3.0)", "(MIT); (GPL-3.0)"),
])
def test_semicolon_column_only_replaces_license_and(expression, expected):
    assert format_concluded_license_semicolon(expression) == expected


@pytest.mark.parametrize(("expression", "expected"), [
    ("MIT", "低风险"),
    ("MIT AND GPL-3.0-only", "低风险;高风险"),
    ("MIT OR GPL-3.0-only", "低风险"),
    ("(MIT OR GPL-3.0-only) AND LGPL-2.1", "低风险;中风险"),
    ("MIT OR GPL-3.0-only AND LGPL-2.1", "低风险;中风险"),
    ("GPL-3.0-only OR (MIT AND LGPL-2.1)", "低风险;中风险"),
    ("(MIT AND GPL-3.0-only) OR (Apache-2.0 AND MPL-2.0)", "低风险;中风险"),
    ("GPL-2.0-only WITH Classpath-exception-2.0 AND MIT", "高风险;低风险"),
    ("MIT AND Others", "低风险"),
    ("MIT AND SomeWeirdLicense", "低风险;未知风险"),
    ("MIT OR SomeWeirdLicense", "低风险"),
    ("Others", "未知风险"),
    ("Unlicensed", "高风险"),
])
def test_output_risk_lists_selected_licenses_in_and_order(expression, expected):
    assert get_output_risk_levels(expression) == expected


def test_api_output_places_new_column_after_original_expression():
    output = _generate_output([
        {"input_name": "demo", "license_type": "MIT AND GPL-3.0-only", "status": "success"},
        {"input_name": "other", "license_type": "MIT", "status": "success",
         "license_analysis": {"thirdparty_dirs": ["vendor"]}},
    ])

    assert output.columns.get_loc("concluded_license_semicolon") == output.columns.get_loc("concluded_license") + 1
    assert output["concluded_license"].tolist() == ["MIT AND GPL-3.0-only", "MIT AND Others"]
    assert output["concluded_license_semicolon"].tolist() == ["MIT; GPL-3.0-only", "MIT AND Others"]
    assert output["risk_level"].tolist() == ["低风险;高风险", "低风险"]


@pytest.mark.asyncio
async def test_cli_excel_has_new_column_and_separate_risks(monkeypatch, tmp_path):
    import main

    async def fake_initialize_api():
        return object()

    async def fake_process_all_repos(api, df, max_concurrency):
        return [{"input_name": "demo", "license_type": "(MIT OR GPL-3.0-only) AND LGPL-2.1",
                 "status": "success"}]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "initialize_api", fake_initialize_api)
    monkeypatch.setattr(main, "process_all_repos", fake_process_all_repos)
    monkeypatch.setattr(main.pd, "read_excel", lambda path: pd.DataFrame([{}]))

    await main.main_async()

    workbook = load_workbook(tmp_path / "outputs" / "output_latest.xlsx", read_only=True)
    rows = list(workbook["分析结果"].values)
    headers = rows[0]
    assert headers.index("concluded_license_semicolon") == headers.index("concluded_license") + 1
    assert rows[1][headers.index("concluded_license")] == "(MIT OR GPL-3.0-only) AND LGPL-2.1"
    assert rows[1][headers.index("concluded_license_semicolon")] == "(MIT OR GPL-3.0-only); LGPL-2.1"
    assert rows[1][headers.index("risk_level")] == "低风险;中风险"
