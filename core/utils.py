def is_sha_version(version: str) -> bool:
    """
    判断字符串是否为Git提交SHA（7~40位十六进制，无点号）。
    避免误判如'1.2.3'、'202406'等常规版本号。
    """
    if not isinstance(version, str):
        return False
    v = version.strip()
    # 只允许全十六进制且长度7~40且不包含点号
    return (
        7 <= len(v) <= 40 and
        '.' not in v and
        v.lower() == v and  # git sha 通常为小写
        all(c in '0123456789abcdef' for c in v)
    )