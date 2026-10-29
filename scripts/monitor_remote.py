"""Keep the local dashboard available while its SSH data connection reconnects."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gzip
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener

SCRIPTS = str(Path(__file__).resolve().parent)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from monitor_training import ThreadingHTTPServer, make_handler


class RemoteStore:
    source_label = "4090 服务器"
    unavailable_message = "服务器连接中断，正在自动重连。页面仍可使用，已显示的数据会保留。"

    def __init__(self, port=6012):
        self.base_url = f"http://127.0.0.1:{port}"
        self.opener = build_opener(ProxyHandler({}))
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.last_success_at = None
        self.last_error_at = None
        self.last_error = None
        self.tunnel = None

    def success(self):
        with self.lock:
            self.last_success_at = time.time()
            self.last_error = None

    def get(self, path, timeout):
        with self.lock:
            cached = self.cache.get(path)
        headers = {"Accept-Encoding": "gzip"}
        if cached:
            headers["If-None-Match"] = cached[0]
        try:
            with self.opener.open(Request(self.base_url + path, headers=headers), timeout=timeout) as response:
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ValueError("Monitor response exceeds 16 MiB")
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                data = json.loads(body)
                etag = response.headers.get("ETag")
        except HTTPError as exc:
            if exc.code == 304 and cached:
                self.success()
                return cached[1]
            if exc.code == 404:
                raise KeyError(path) from exc
            with self.lock:
                self.last_error_at, self.last_error = time.time(), f"HTTP {exc.code}"
            raise
        except (OSError, ValueError) as exc:
            with self.lock:
                self.last_error_at, self.last_error = time.time(), type(exc).__name__
            raise
        self.success()
        if etag:
            with self.lock:
                self.cache[path] = (etag, data)
                self.cache.move_to_end(path)
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
        return data

    def discover(self):
        data = self.get("/api/runs", 8)
        return {run["id"]: run for run in data["runs"]}

    def load(self, ident):
        return self.get("/api/run?id=" + quote(ident, safe=""), 30)

    def catalog(self):
        return self.get("/api/catalog", 30)

    def hands(self, ident, trace_id, **filters):
        query = {"id": ident, "trace": trace_id, **filters}
        query["category"] = query.pop("category_filter", "all")
        return self.get("/api/hands?" + urlencode(query), 35)

    def hand(self, ident, trace_id, record):
        return self.get("/api/hand?" + urlencode({"id": ident, "trace": trace_id, "record": record}), 15)

    def health(self):
        process = self.tunnel.process if self.tunnel else None
        with self.lock:
            return {"page": "available", "source": self.source_label,
                "ssh_process": "running" if process and process.poll() is None else "reconnecting",
                "last_success_at": self.last_success_at, "last_error_at": self.last_error_at,
                "last_error": self.last_error, "checked_at": time.time(),
                "process_liveness": "unknown"}


class Tunnel:
    def __init__(self, host, local_port, remote_port):
        self.command = ["/usr/bin/ssh", "-NT", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=10",
                        "-o", "ServerAliveCountMax=2", "-L",
                        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", host]
        self.stopping = threading.Event()
        self.process = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        backoff = 1
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL)
                self.process = process
                print(f"SSH started pid={process.pid}", flush=True)
                while process.poll() is None and not self.stopping.wait(.2):
                    pass
                if self.stopping.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return
                if time.monotonic() - started >= 60:
                    backoff = 1
                print(f"SSH exited status={process.returncode}; retry in {backoff}s", flush=True)
            except OSError as exc:
                print(f"SSH unavailable: {exc}; retry in {backoff}s", flush=True)
            if self.stopping.wait(backoff):
                return
            backoff = min(backoff * 2, 15)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--tunnel-port", type=int, default=6012)
    parser.add_argument("--remote-port", type=int, default=6008)
    parser.add_argument("--ssh-host", default="seetacloud-poker")
    parser.add_argument("--open-browser", action="store_true", help="open the local dashboard after binding")
    args = parser.parse_args()
    # Bind the stable page before connecting SSH. Never take another listener's port.
    store = RemoteStore(args.tunnel_port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    tunnel = Tunnel(args.ssh_host, args.tunnel_port, args.remote_port)
    store.tunnel = tunnel
    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    tunnel.thread.start()
    print(f"Overwatch local gateway: http://127.0.0.1:{args.port}", flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.close()
        server.server_close()


if __name__ == "__main__":
    main()
