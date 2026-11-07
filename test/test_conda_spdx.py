"""The one approved same-version only/later rule never becomes a general intersection."""
import pytest
from core.conda_spdx import narrow_metadata_licenses,reduce_only_or_later
from core.utils import format_concluded_license_semicolon,get_output_risk_levels


@pytest.mark.parametrize('values,expected',[
    (['GPL-3.0-only','GPL-3.0-or-later'],['GPL-3.0-only']),
    (['GPL-3.0-or-later','GPL-3.0-only','MIT'],['GPL-3.0-only','MIT']),
    (['LGPL-2.1-only','LGPL-2.1-or-later'],['LGPL-2.1-only']),
    (['AGPL-3.0-only','AGPL-3.0-or-later'],['AGPL-3.0-only']),
    (['GPL-2.0-or-later','GPL-3.0-or-later'],['GPL-2.0-or-later','GPL-3.0-or-later']),
    (['GPL-2.0-only','GPL-3.0-only'],['GPL-2.0-only','GPL-3.0-only']),
    (['GPL-3.0-only','LGPL-3.0-or-later'],['GPL-3.0-only','LGPL-3.0-or-later']),
    (['MIT','BSD-3-Clause','MIT'],['MIT','BSD-3-Clause']),
    (['GPL-3.0-only OR MIT','GPL-3.0-or-later'],['GPL-3.0-only OR MIT','GPL-3.0-or-later']),
    (['GPL-3.0-only WITH Classpath-exception-2.0','GPL-3.0-or-later'],['GPL-3.0-only WITH Classpath-exception-2.0','GPL-3.0-or-later']),
])
def test_standalone_metadata_rule(values,expected):
    original=values[:]
    assert narrow_metadata_licenses(values)==expected
    assert values==original


@pytest.mark.parametrize('expression,expected',[
    ('GPL-3.0-only AND GPL-3.0-or-later','GPL-3.0-only'),
    ('GPL-3.0-or-later AND (GPL-3.0-only AND MIT)','GPL-3.0-only AND MIT'),
    ('(GPL-3.0-only AND GPL-3.0-or-later) OR MIT','GPL-3.0-only OR MIT'),
    ('(GPL-3.0-only OR MIT) AND GPL-3.0-or-later','(GPL-3.0-only OR MIT) AND GPL-3.0-or-later'),
    ('GPL-3.0-only OR GPL-3.0-or-later','GPL-3.0-only OR GPL-3.0-or-later'),
    ('GPL-2.0-or-later AND GPL-3.0-or-later','GPL-2.0-or-later AND GPL-3.0-or-later'),
    ('GPL-2.0-only AND GPL-3.0-only','GPL-2.0-only AND GPL-3.0-only'),
    ('GPL-3.0-only AND LGPL-3.0-or-later','GPL-3.0-only AND LGPL-3.0-or-later'),
    ('GPL-3.0-only WITH Classpath-exception-2.0 AND GPL-3.0-or-later','GPL-3.0-only WITH Classpath-exception-2.0 AND GPL-3.0-or-later'),
    ('MIT AND (Apache-2.0 OR BSD-3-Clause)','MIT AND (Apache-2.0 OR BSD-3-Clause)'),
    ('MIT OR Apache-2.0 AND BSD-3-Clause','MIT OR Apache-2.0 AND BSD-3-Clause'),
    ('MIT AND (MIT AND BSD-3-Clause)','MIT AND BSD-3-Clause'),
    ('LicenseRef-Custom AND MIT','LicenseRef-Custom AND MIT'),
])
def test_spdx_structure_preserved(expression,expected):
    assert reduce_only_or_later(expression)==expected


@pytest.mark.parametrize('bad',['','MIT BSD-3-Clause','MIT AND','(MIT OR BSD-3-Clause','MIT)','MIT WITH','MIT WITH AND','MIT ; BSD-3-Clause'])
def test_invalid_expression_is_not_flattened(bad):
    with pytest.raises(ValueError):reduce_only_or_later(bad)


def test_existing_display_and_risk_use_final_expression():
    final=reduce_only_or_later('GPL-3.0-or-later AND GPL-3.0-only AND (MIT OR BSD-3-Clause)')
    assert final=='GPL-3.0-only AND (MIT OR BSD-3-Clause)'
    assert format_concluded_license_semicolon(final)=='GPL-3.0-only; MIT OR BSD-3-Clause'
    assert get_output_risk_levels(final)=='高风险;低风险'
