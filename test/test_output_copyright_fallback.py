import pandas as pd
import pytest
from openpyxl import load_workbook

from api import _generate_output


def test_final_output_fills_only_blank_copyright_notices():
    results = [
        {"input_name": "alpha", "copyright_notice": None, "status": "success"},
        {"input_name": "beta", "copyright_notice": "  ", "status": "error"},
        {"input_name": "gamma", "copyright_notice": float("nan"), "status": "success"},
        {"input_name": "delta", "copyright_notice": "Copyright 2020 Delta", "status": "success"},
    ]

    output = _generate_output(results)

    assert output["copyright_notice"].tolist() == [
        "Copyright alpha Original author and authors",
        "Copyright beta Original author and authors",
        "Copyright gamma Original author and authors",
        "Copyright 2020 Delta",
    ]
    assert output["status"].tolist() == ["success", "error", "success", "success"]


def test_final_output_adds_copyright_column_when_all_results_omit_it():
    output = _generate_output([{"input_name": "epsilon", "status": "error"}])

    assert output.loc[0, "copyright_notice"] == "Copyright epsilon Original author and authors"
    assert output.columns.get_loc("copyright_notice") == output.columns.get_loc("risk_level") + 1


@pytest.mark.asyncio
async def test_cli_excel_uses_same_final_copyright_check(monkeypatch, tmp_path):
    import main

    async def fake_initialize_api():
        return object()

    async def fake_process_all_repos(api, df, max_concurrency):
        return [
            {"input_name": "alpha", "copyright_notice": None, "status": "success"},
            {"input_name": "delta", "copyright_notice": "Copyright 2020 Delta", "status": "success"},
        ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "initialize_api", fake_initialize_api)
    monkeypatch.setattr(main, "process_all_repos", fake_process_all_repos)
    monkeypatch.setattr(main.pd, "read_excel", lambda path: pd.DataFrame([{}, {}]))

    await main.main_async()

    workbook = load_workbook(tmp_path / "outputs" / "output_latest.xlsx", read_only=True)
    rows = list(workbook["分析结果"].values)
    notice_col = rows[0].index("copyright_notice")
    assert [row[notice_col] for row in rows[1:]] == [
        "Copyright alpha Original author and authors",
        "Copyright 2020 Delta",
    ]
