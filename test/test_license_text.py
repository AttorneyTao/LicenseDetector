# -*- coding: utf-8 -*-
"""Tests for the license_text output column (full license text retention)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import prepare_license_text
from core.config import RESULT_COLUMNS_ORDER


class TestPrepareLicenseText:
    def test_plain_text_passthrough(self):
        text = "MIT License\n\nCopyright (c) 2024 Foo"
        assert prepare_license_text(text) == text

    def test_none_and_empty(self):
        assert prepare_license_text(None) is None
        assert prepare_license_text("") is None
        assert prepare_license_text(123) is None

    def test_strips_excel_illegal_control_chars(self):
        text = "MIT\x00 License\x0b with\x1f junk"
        cleaned = prepare_license_text(text)
        assert cleaned == "MIT License with junk"

    def test_keeps_newlines_and_tabs(self):
        text = "line1\nline2\tindent\r\nline3"
        assert prepare_license_text(text) == text

    def test_truncates_over_excel_cell_limit(self):
        text = "A" * 40000
        cleaned = prepare_license_text(text)
        assert len(cleaned) <= 32767  # Excel cell hard limit
        assert cleaned.endswith("...[TRUNCATED]")

    def test_column_in_output_order(self):
        assert "license_text" in RESULT_COLUMNS_ORDER


class TestOutputColumnPresence:
    def test_api_generate_output_always_has_license_text(self):
        from api import _generate_output
        # 结果中不带 license_text 字段时，输出也必须有这一列
        results = [{
            "input_url": "https://example.com", "status": "error",
            "license_type": None, "readme_license": None, "license_file_license": None,
        }]
        df = _generate_output(results)
        assert "license_text" in df.columns

    def test_api_generate_output_preserves_text(self):
        from api import _generate_output
        text = "Apache License\nVersion 2.0, January 2004\n..."
        results = [{
            "input_url": "https://github.com/foo/bar", "status": "success",
            "license_type": "Apache-2.0", "readme_license": None,
            "license_file_license": "Apache-2.0", "license_text": text,
        }]
        df = _generate_output(results)
        assert df.iloc[0]["license_text"] == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
