from core.github_utils import find_top_level_thirdparty_dirs

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