import pytest
import asyncio
from core.go_utils import get_github_url_from_pkggo

@pytest.mark.asyncio
@pytest.mark.parametrize("pkggo_url,version,expected_github", [
    # 真实数据，实际接口调用
    ("https://pkg.go.dev/go.uber.org/atomic", "1.9.0", "https://github.com/uber-go/atomic"),
    ("https://pkg.go.dev/golang.org/x/exp", "0.0.0-20240325151524-a685a6edb6d8", "https://github.com/golang/exp"),
    ("https://pkg.go.dev/golang.org/x/crypto", "0.30.0", "https://github.com/golang/crypto"),
    ("https://pkg.go.dev/gopkg.in/yaml.v3", "3.0.1", "https://github.com/go-yaml/yaml"),
    ("https://pkg.go.dev/go.opentelemetry.io/otel", "1.36.0", "https://github.com/open-telemetry/opentelemetry-go"),
])
async def test_get_github_url_from_pkggo_real(pkggo_url, version, expected_github):
    result = await get_github_url_from_pkggo(pkggo_url, version)
    print(result)
    assert result["github_url"] is not None
    assert expected_github in result["github_url"]