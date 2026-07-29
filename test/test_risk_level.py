"""风险等级聚合的单元测试。

重点覆盖 SPDX 运算符语义：
- AND 取最高风险、OR 取最低风险、AND 优先级高于 OR
- WITH 的例外条款不改变许可证本身的风险
- "Others" 结构性标记不参与聚合
"""

import pytest

from core.utils import (
    _load_risk_config,
    evaluate_spdx_risk,
    get_risk_level,
)


@pytest.fixture(scope="module")
def labels():
    return _load_risk_config()["labels"]


# ---------------------------------------------------------------------------
# 单个许可证
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("MIT", "low"),
    ("Apache-2.0", "low"),
    ("LGPL-2.1", "medium"),
    ("MPL-2.0", "medium"),
    ("GPL-3.0-only", "high"),
    ("AGPL-3.0-or-later", "high"),
    ("Unlicensed", "high"),
    ("SomeWeirdLicense", "unknown"),
])
def test_single_license(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


@pytest.mark.parametrize("expr,expected", [
    ("GPL-3.0", "high"),
    ("GPL-3.0+", "high"),
    ("GPL-3.0-only", "high"),
    ("GPL-3.0-or-later", "high"),
    ("gpl-3.0-ONLY", "high"),
])
def test_spdx_suffix_and_case_normalization(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


# ---------------------------------------------------------------------------
# AND：取最高风险
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("MIT AND Apache-2.0", "low"),
    ("MIT AND LGPL-2.1", "medium"),
    ("MIT AND GPL-3.0-or-later", "high"),
    ("LGPL-2.1 AND GPL-2.0-only", "high"),
    ("MIT AND Apache-2.0 AND AGPL-3.0", "high"),
    ("mit and gpl-3.0-only", "high"),  # 运算符大小写不敏感
])
def test_and_takes_highest_risk(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


# ---------------------------------------------------------------------------
# OR：取最低风险
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("MIT OR GPL-3.0-or-later", "low"),
    ("GPL-3.0-or-later OR MIT", "low"),
    ("LGPL-2.1 OR MPL-2.0", "medium"),
    ("AGPL-3.0 OR LGPL-2.1", "medium"),
    ("GPL-2.0-only OR AGPL-3.0", "high"),
    ("GPL-3.0-only OR LGPL-2.1 OR MIT", "low"),
])
def test_or_takes_lowest_risk(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


def test_or_prefers_determinate_branch_over_unknown():
    # 无法判定的分支不能作为"最低风险"的依据
    assert evaluate_spdx_risk("MIT OR SomeWeirdLicense") == "low"
    assert evaluate_spdx_risk("GPL-2.0-only OR SomeWeirdLicense") == "high"


def test_or_all_unknown_stays_unknown():
    assert evaluate_spdx_risk("WeirdOne OR WeirdTwo") == "unknown"


# ---------------------------------------------------------------------------
# 优先级与括号
# ---------------------------------------------------------------------------

def test_and_binds_tighter_than_or():
    # MIT OR (GPL-3.0 AND AGPL-3.0) -> low
    assert evaluate_spdx_risk("MIT OR GPL-3.0-only AND AGPL-3.0") == "low"
    # (MIT AND GPL-3.0) OR LGPL-2.1 -> medium
    assert evaluate_spdx_risk("MIT AND GPL-3.0-only OR LGPL-2.1") == "medium"


@pytest.mark.parametrize("expr,expected", [
    ("GPL-3.0-only OR (MIT AND LGPL-2.1-only)", "medium"),
    ("(MIT OR GPL-3.0-only) AND Apache-2.0", "low"),
    ("(MIT OR GPL-3.0-only) AND LGPL-2.1", "medium"),
    ("((MIT))", "low"),
    ("Apache-2.0 AND (MIT AND BSD-3-Clause)", "low"),
])
def test_parentheses(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


# ---------------------------------------------------------------------------
# WITH：例外条款不改变风险
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("Apache-2.0 WITH LLVM-exception", "low"),
    ("GPL-2.0-only WITH Classpath-exception-2.0", "high"),
    ("Apache-2.0 WITH LLVM-exception OR MIT", "low"),
    ("Apache-2.0 WITH LLVM-exception AND GPL-3.0-only", "high"),
])
def test_with_exception_keeps_license_risk(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


# ---------------------------------------------------------------------------
# Others：结构性标记，不参与聚合
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("MIT AND Others", "low"),
    ("Apache-2.0 AND Others", "low"),
    ("LGPL-2.1 AND Others", "medium"),
    ("GPL-3.0-only AND Others", "high"),
    ("MIT OR GPL-3.0-only AND Others", "low"),
    ("Apache-2.0 AND (MIT AND BSD-3-Clause) AND Others", "low"),
    ("others and MIT", "low"),  # 位置与大小写都不影响
])
def test_others_marker_is_ignored(expr, expected):
    assert evaluate_spdx_risk(expr) == expected


def test_others_alone_is_undetermined():
    # 没有"之前的表达式"可依据，只能是未知
    assert evaluate_spdx_risk("Others") is None
    assert get_risk_level("Others") == _load_risk_config()["labels"]["unknown"]


def test_others_does_not_change_result_of_any_expression():
    for expr in ["MIT", "GPL-3.0-only", "LGPL-2.1", "MIT OR GPL-3.0-only",
                 "MIT AND GPL-3.0-only", "SomeWeirdLicense"]:
        assert evaluate_spdx_risk(f"{expr} AND Others") == evaluate_spdx_risk(expr), expr


# ---------------------------------------------------------------------------
# 空值与畸形表达式
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
def test_empty_values_are_unknown(value, labels):
    assert get_risk_level(value) == labels["unknown"]


@pytest.mark.parametrize("expr,expected", [
    ("MIT AND (", "low"),          # 括号不匹配，降级为取最高风险
    ("MIT AND GPL-3.0-only)", "high"),
    ("AND MIT", "low"),
    ("MIT OR", "low"),
])
def test_malformed_expression_falls_back_to_worst_risk(expr, expected):
    # 降级路径必须仍然给出结果，不能抛异常中断整批处理
    assert evaluate_spdx_risk(expr) == expected


# ---------------------------------------------------------------------------
# get_risk_level 的标签输出
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,level_key", [
    ("MIT", "low"),
    ("LGPL-2.1", "medium"),
    ("GPL-3.0-only", "high"),
    ("MIT OR GPL-3.0-only", "low"),
    ("MIT AND GPL-3.0-only", "high"),
    ("MIT AND Others", "low"),
])
def test_get_risk_level_returns_configured_label(expr, level_key, labels):
    assert get_risk_level(expr) == labels[level_key]
