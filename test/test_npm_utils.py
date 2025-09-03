import os
import tarfile
import tempfile
import shutil
import pytest
from core.npm_utils import async_analyze_npm_tarball_thirdparty_dirs

@pytest.mark.asyncio
async def test_async_analyze_npm_tarball_thirdparty_dirs(monkeypatch):
    # 1. 创建一个 fake tarball，包含 third_party 和 vendor 目录
    tmp_dir = tempfile.mkdtemp()
    package_dir = os.path.join(tmp_dir, "package")
    os.makedirs(os.path.join(package_dir, "third_party"))
    os.makedirs(os.path.join(package_dir, "vendor"))
    with open(os.path.join(package_dir, "third_party", "foo.txt"), "w") as f:
        f.write("test")
    tarball_path = os.path.join(tmp_dir, "fake.tgz")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(package_dir, arcname="package")

    # 2. mock aiohttp 下载，直接读取本地 fake.tgz
    import aiohttp
    import aiofiles

    class FakeResponse:
        def __init__(self, path):
            self.path = path
            self.status = 200
            self.content = self
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        def raise_for_status(self): pass
        async def iter_chunked(self, chunk_size):
            async with aiofiles.open(self.path, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        def get(self, url):  # 这里改为普通def
            return FakeResponse(tarball_path)

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: FakeSession())

    # 3. 调用异步分析函数
    dirs = await async_analyze_npm_tarball_thirdparty_dirs("http://fake-url/fake.tgz")
    assert "third_party" in dirs
    assert "vendor" in dirs

    # 4. 清理
    shutil.rmtree(tmp_dir)

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

