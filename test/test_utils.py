from core.utils import extract_thirdparty_dirs_column, find_top_level_thirdparty_dirs
import pandas as pd

def test_extract_thirdparty_dirs_column():
    data = [
        {"license_analysis": {"thirdparty_dirs": ["third_party", "src/thirdparty"]}},
        {"license_analysis": {"thirdparty_dirs": []}},
        {"license_analysis": {}},
        {"license_analysis": {"thirdparty_dirs": ["deps/third-party"]}},
        {"license_analysis": None},
    ]
    df = pd.DataFrame(data)
    result_df = extract_thirdparty_dirs_column(df)
    assert result_df.loc[0, "thirdparty_dirs"] == "本项目包含第三方组件，请关注：third_party,src/thirdparty 目录"
    assert result_df.loc[1, "thirdparty_dirs"] == ""
    assert result_df.loc[2, "thirdparty_dirs"] == ""
    assert result_df.loc[3, "thirdparty_dirs"] == "本项目包含第三方组件，请关注：deps/third-party 目录"
    assert result_df.loc[4, "thirdparty_dirs"] == ""


def test_find_top_level_thirdparty_dirs():
    tree = [
        {"type": "tree", "path": "third_party"},
        {"type": "tree", "path": "third_party/foo"},
        {"type": "tree", "path": "src/thirdparty"},
        {"type": "tree", "path": "src/thirdparty/bar"},
        {"type": "tree", "path": "deps/third-party"},
        {"type": "tree", "path": "deps/third-party/baz"},
        {"type": "tree", "path": "thirdparty"},
        {"type": "tree", "path": "thirdparty/foo/bar"},
        {"type": "tree", "path": "not_thirdparty"},
        {"type": "blob", "path": "thirdparty/LICENSE"},
    ]
    result = find_top_level_thirdparty_dirs(tree)
    assert set(result) == {"third_party", "src/thirdparty", "deps/third-party", "thirdparty"}