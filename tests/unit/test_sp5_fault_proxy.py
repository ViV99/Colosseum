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


def test_a_slow_receiver_backpressures_instead_of_cutting():
    """A target that does not read for a while (a backpressuring learner) only slows the sender down: nothing is cut,
    every byte arrives (review round 1: a socket timeout used to turn a ``sendall`` blocked for 50 ms into a cut)."""
    total = 16 * 2**20                                            # more than the socket buffers on the way hold
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 18)
    server.bind(("127.0.0.1", 0))
    server.listen()
    got = []
    paused = threading.Event()

    def sink():
        conn, _ = server.accept()
        with conn:
            paused.wait(10)                                       # nothing read until the proxy is stuck a while
            n = 0
            while n < total:
                data = conn.recv(1 << 16)
                if not data:
                    break
                n += len(data)
            got.append(n)

    reader = threading.Thread(target=sink, daemon=True)
    reader.start()
    try:
        with TcpFaultProxy(("127.0.0.1", server.getsockname()[1])) as proxy:
            def release():
                while proxy.bytes_forwarded == 0 and not paused.is_set():
                    time.sleep(0.01)
                time.sleep(0.3)                                   # the proxy's sendall waits far beyond 50 ms
                paused.set()

            threading.Thread(target=release, daemon=True).start()
            with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
                s.sendall(b"x" * total)
                reader.join(15)
            assert got == [total] and proxy.bytes_forwarded == total
    finally:
        paused.set()
        server.close()


def _pipes_alive():
    return [t for t in threading.enumerate() if t.name == "fault-proxy-pipe"]


def test_a_pipe_blocked_in_sendall_ends_when_the_other_direction_ends():
    """The target half-closes without reading while the client's bytes are stuck in the proxy's ``sendall``: the
    reverse pipe ends at the EOF and shuts both sockets down, which wakes the blocked ``sendall`` (``close`` alone
    does not on Linux)."""
    server = socket.create_server(("127.0.0.1", 0))
    accepted = []

    def target():
        conn, _ = server.accept()
        accepted.append(conn)                                      # never reads

    threading.Thread(target=target, daemon=True).start()
    before = len(_pipes_alive())
    with TcpFaultProxy(("127.0.0.1", server.getsockname()[1])) as proxy:
        client = socket.create_connection(("127.0.0.1", proxy.port), timeout=10)

        def flood():
            try:
                client.sendall(b"x" * (64 * 2**20))
            except OSError:
                pass

        sender = threading.Thread(target=flood, daemon=True)
        sender.start()
        deadline = time.monotonic() + 10
        while (not accepted or proxy.bytes_forwarded == 0) and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)                                            # the proxy's sendall is blocked by now
        accepted[0].shutdown(socket.SHUT_WR)                       # EOF towards the client, still not reading
        deadline = time.monotonic() + 5
        while len(_pipes_alive()) > before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(_pipes_alive()) == before                       # both pipes ended, none stuck in sendall
        sender.join(5)
        client.close()
    accepted[0].close()
    server.close()
