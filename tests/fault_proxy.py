"""TCP fault injection between two local endpoints (SP5 T3.1, used by T3.1 and T3.2; test support, imported by bare
name).

``TcpFaultProxy(("127.0.0.1", port))`` listens on ``127.0.0.1:<ephemeral>`` (``proxy.address``) and forwards
every accepted connection to the target, both directions, through its own threads. The target may be set later
(``TcpFaultProxy()`` then ``set_target``: a node can announce the proxy's address before its own data port is
bound); until then connections are closed at once. Faults:
- ``delay(sec)``: every forwarded read waits ``sec`` first (both directions);
- ``cut(sec)``: every open connection is closed now and new ones are accepted and closed at once for ``sec``
  seconds (``time.monotonic``), then forwarding resumes by itself;
- ``drop()``: like ``cut`` without an end (until ``restore()``);
- ``stall()``: connections are accepted but nothing is forwarded (a TLS handshake never completes: the
  "unreachable within connect_timeout_sec" case of spec 4.8) until ``restore()``;
- ``restore()``: normal forwarding (also ends a delay).
``connections_seen`` counts accepted connections, ``bytes_forwarded`` the bytes moved. ``close()`` stops
everything; use it as a context manager. No wall clock, no sleeps longer than 50 ms per loop (besides ``delay``).

The forwarding sockets stay blocking: a pipe waits for data with ``select`` (50 ms slices, so it notices ``close``)
and forwards it with a blocking ``sendall``, so a slow receiver backpressures the sender through the proxy as it
would on a direct connection (a timeout on the socket would also apply to the other direction's ``sendall`` and
turn backpressure into a cut). ``shutdown`` in a cut unblocks a stuck ``sendall``.
"""

from __future__ import annotations

import select
import socket
import threading
import time

_POLL_SEC = 0.05
_BUF = 1 << 16


class TcpFaultProxy:
    def __init__(self, target: tuple[str, int] | None = None, *, host: str = "127.0.0.1") -> None:
        self._target = target
        self._lock = threading.Lock()
        self._mode = "normal"            # normal | cut | stall
        self._cut_until = 0.0
        self._delay = 0.0
        self._conns: list[socket.socket] = []
        self._closed = threading.Event()
        self.connections_seen = 0
        self.bytes_forwarded = 0
        self._server = socket.create_server((host, 0))
        self._server.settimeout(_POLL_SEC)
        self.port = self._server.getsockname()[1]
        self.address = f"{host}:{self.port}"
        self._acceptor = threading.Thread(target=self._accept_loop, name="fault-proxy", daemon=True)
        self._acceptor.start()

    def set_target(self, target: tuple[str, int]) -> None:
        with self._lock:
            self._target = target

    # --- faults ------------------------------------------------------------------------------------------------

    def delay(self, sec: float) -> None:
        with self._lock:
            self._delay = float(sec)

    def cut(self, sec: float) -> None:
        with self._lock:
            self._mode, self._cut_until = "cut", time.monotonic() + float(sec)
        self._close_all()

    def drop(self) -> None:
        with self._lock:
            self._mode, self._cut_until = "cut", float("inf")
        self._close_all()

    def stall(self) -> None:
        with self._lock:
            self._mode = "stall"
        self._close_all()

    def restore(self) -> None:
        with self._lock:
            self._mode, self._cut_until, self._delay = "normal", 0.0, 0.0

    def close(self) -> None:
        self._closed.set()
        self._close_all()
        self._server.close()
        self._acceptor.join(timeout=2)

    def __enter__(self) -> TcpFaultProxy:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- internals -----------------------------------------------------------------------------------------------

    def _state(self) -> tuple[str, float]:
        with self._lock:
            if self._mode == "cut" and time.monotonic() >= self._cut_until:
                self._mode = "normal"
            return self._mode, self._delay

    def _close_all(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        for s in conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections_seen += 1                 # only this thread writes it
            client.settimeout(None)
            mode, _ = self._state()
            with self._lock:
                target = self._target
            if mode == "cut" or target is None:
                client.close()
                continue
            if mode == "stall":
                with self._lock:
                    self._conns.append(client)          # held open, never forwarded
                continue
            try:
                upstream = socket.create_connection(target, timeout=2.0)
                upstream.settimeout(None)                # blocking from now on (module docstring)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._conns += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(a, b), name="fault-proxy-pipe", daemon=True).start()

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while not self._closed.is_set():
                ready, _, _ = select.select([src], [], [], _POLL_SEC)
                if not ready:
                    continue
                data = src.recv(_BUF)
                if not data:
                    break
                mode, delay = self._state()
                if mode != "normal":
                    break
                if delay > 0:
                    time.sleep(delay)
                dst.sendall(data)
                with self._lock:
                    self.bytes_forwarded += len(data)
        except (OSError, ValueError):                    # closed by a cut (ValueError: select on a closed socket)
            pass
        finally:
            with self._lock:
                self._conns = [c for c in self._conns if c is not src and c is not dst]
            for s in (src, dst):                       # shutdown wakes the other pipe's blocked sendall / select
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
