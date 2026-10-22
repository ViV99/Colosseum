"""TcpFaultProxy (SP5 T3.1 test support): forwards, delays, cuts for a while, drops, stalls and restores."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from fault_proxy import TcpFaultProxy


@pytest.fixture
def echo():
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.05)
    stop = threading.Event()

    def serve(conn):
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                conn.sendall(data)

    def loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:                                          # closed at teardown
                return
            threading.Thread(target=serve, args=(conn,), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    yield server.getsockname()[1]
    stop.set()
    server.close()


def _roundtrip(port, payload=b"ping", timeout=1.0):
    """The echoed bytes; b"" when the proxy closed the connection; None on a refusal, a reset or a timeout."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall(payload)
            return s.recv(4096)
    except OSError:
        return None


def test_forwards_and_counts(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        assert _roundtrip(proxy.port) == b"ping"
        assert proxy.connections_seen == 1 and proxy.bytes_forwarded >= 8


def test_cut_closes_connections_and_ends_by_itself(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        s = socket.create_connection(("127.0.0.1", proxy.port), timeout=1.0)
        s.sendall(b"a")
        assert s.recv(10) == b"a"
        proxy.cut(0.5)
        assert s.recv(10) == b""                                     # the open connection was closed
        s.close()
        assert not _roundtrip(proxy.port)                            # closed at once while the cut lasts
        time.sleep(0.6)
        assert _roundtrip(proxy.port) == b"ping"                     # and restored by itself


def test_drop_and_stall_last_until_restore(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        proxy.drop()
        assert not _roundtrip(proxy.port)
        proxy.stall()
        assert _roundtrip(proxy.port, timeout=0.3) is None            # accepted, never answered: timeout
        proxy.restore()
        assert _roundtrip(proxy.port) == b"ping"


def test_delay_slows_every_read(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        proxy.delay(0.2)
        t0 = time.monotonic()
        assert _roundtrip(proxy.port) == b"ping"
        assert time.monotonic() - t0 >= 0.35                         # two forwarded reads, 0.2 s each


def test_a_target_set_later_is_used_from_then_on(echo):
    with TcpFaultProxy() as proxy:
        assert not _roundtrip(proxy.port)                            # no target yet: closed at once
        proxy.set_target(("127.0.0.1", echo))
        assert _roundtrip(proxy.port) == b"ping"
