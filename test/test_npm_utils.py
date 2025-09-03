import os
import tempfile
import tarfile
import shutil
from core.npm_utils import analyze_npm_tarball_thirdparty_dirs

def create_fake_tarball_with_thirdparty(tmp_dir):
    # 创建一个临时目录结构
    package_dir = os.path.join(tmp_dir, "package")
    os.makedirs(os.path.join(package_dir, "third_party"))
    os.makedirs(os.path.join(package_dir, "vendor"))
    # 创建一个空文件
    with open(os.path.join(package_dir, "third_party", "foo.txt"), "w") as f:
        f.write("test")
    # 打包成 tar.gz
    tarball_path = os.path.join(tmp_dir, "fake.tgz")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(package_dir, arcname="package")
    return tarball_path

def test_analyze_npm_tarball_thirdparty_dirs_local():
    tmp_dir = tempfile.mkdtemp()
    try:
        tarball_path = create_fake_tarball_with_thirdparty(tmp_dir)
        # 用 file:// 协议模拟 requests.get
        import requests
        from unittest.mock import patch

        class FakeResponse:
            def __init__(self, path):
                self.path = path
                self.status_code = 200
            def raise_for_status(self):
                pass
            def iter_content(self, chunk_size=8192):
                with open(self.path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fake_requests_get(url, stream=True):
            # 忽略url，直接返回本地文件
            return FakeResponse(tarball_path)

        with patch("requests.get", fake_requests_get):
            # 这里传入任意字符串即可
            dirs = analyze_npm_tarball_thirdparty_dirs("http://fake-url/fake.tgz")
            # 断言
            assert "third_party" in dirs
            assert "vendor" in dirs
    finally:
        shutil.rmtree(tmp_dir)