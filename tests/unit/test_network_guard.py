"""测试期网络守卫的自测。

守卫本身必须被测：一个**不起作用的守卫比没有守卫更糟**——它会让人以为
"CI 不会打真网"这条规范已经被强制执行，从而放心地写出依赖网络的测试。

第一版守卫就正好踩了这个坑：只放行 loopback，而容器里的出网代理恰好挂在
``127.0.0.1``，于是所有外网请求都从守卫底下溜过去了。这里的用例就是钉死
那个洞。
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from tests.conftest import NetworkAccessInTestError, _is_allowed, _proxy_endpoints


class TestGuardBlocks:
    """拦截行为。"""

    def test_direct_socket_to_public_host_is_blocked(self) -> None:
        with pytest.raises(NetworkAccessInTestError):
            socket.create_connection(("example.com", 443), timeout=1)

    def test_urllib_is_blocked(self) -> None:
        # 有代理时 urllib 会把它包成 URLError，没代理时直接抛我们的异常。
        # 两种都算拦住了，重点是**请求没有真的发出去**
        with pytest.raises((NetworkAccessInTestError, urllib.error.URLError)):
            urllib.request.urlopen("https://example.com", timeout=1)

    def test_loopback_still_allowed(self) -> None:
        # 守卫不能误伤本地服务：TestClient 与本地 uvicorn 用例都要用
        server = socket.create_server(("127.0.0.1", 0))
        try:
            with socket.create_connection(server.getsockname(), timeout=1):
                pass
        finally:
            server.close()


class TestProxyAwareness:
    """代理识别。"""

    def test_proxy_endpoint_is_not_treated_as_local(self) -> None:
        proxies = frozenset({("127.0.0.1", 38487)})
        assert _is_allowed(("127.0.0.1", 38487), proxies) is False

    def test_other_loopback_ports_stay_allowed(self) -> None:
        proxies = frozenset({("127.0.0.1", 38487)})
        assert _is_allowed(("127.0.0.1", 8000), proxies) is True

    def test_public_host_never_allowed(self) -> None:
        assert _is_allowed(("example.com", 443), frozenset()) is False

    @pytest.mark.parametrize(
        "value", ["http://127.0.0.1:38487", "127.0.0.1:38487", "https://proxy.local:3128"]
    )
    def test_proxy_url_forms_are_parsed(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        # 代理变量的写法不统一：有的带 scheme 有的不带
        for var in ("https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", value)

        assert len(_proxy_endpoints()) == 1

    def test_no_proxy_env_yields_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)

        assert _proxy_endpoints() == frozenset()
