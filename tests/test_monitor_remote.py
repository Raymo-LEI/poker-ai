"""The dashboard page survives a broken SSH data connection and recovers."""

import importlib.util
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import threading


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("monitor_remote", SCRIPTS / "monitor_remote.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)
import monitor_training as monitor


def start(store, port=0):
    server = monitor.ThreadingHTTPServer(("127.0.0.1", port), monitor.make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def get(port, path):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_gateway_stays_open_offline_and_recovers_with_cached_etags(tmp_path):
    run = tmp_path / "experiment"
    run.mkdir()
    (run / "metrics.csv").write_text("iteration,p0_loss\n100,12\n")
    upstream, upstream_thread = start(monitor.Store([tmp_path]))
    upstream_port = upstream.server_port
    remote_store = remote.RemoteStore(upstream_port)
    gateway, gateway_thread = start(remote_store)
    try:
        url = "/api/run?id=0/experiment"
        status, body = get(gateway.server_port, url)
        assert status == 200 and json.loads(body)["latest"]["step"] == 100
        cached = remote_store.load("0/experiment")
        assert remote_store.load("0/experiment") is cached  # upstream 304
        assert get(gateway.server_port, "/api/catalog")[0] == 200
        stop(upstream, upstream_thread)
        upstream = None
        assert get(gateway.server_port, "/")[0] == 200
        assert get(gateway.server_port, "/app.js")[0] == 200
        assert get(gateway.server_port, "/workbench.js")[0] == 200
        assert get(gateway.server_port, "/api/health")[0] == 200
        status, body = get(gateway.server_port, "/api/runs")
        assert status == 503 and "自动重连" in json.loads(body)["error"]
        # Do not falsely report a connection using old cached data.
        assert get(gateway.server_port, url)[0] == 503
        assert get(gateway.server_port, "/api/catalog")[0] == 503
        (run / "metrics.csv").write_text("iteration,p0_loss\n101,8\n")
        upstream, upstream_thread = start(monitor.Store([tmp_path]), upstream_port)
        status, body = get(gateway.server_port, url)
        assert status == 200 and json.loads(body)["latest"]["step"] == 101
    finally:
        stop(gateway, gateway_thread)
        if upstream is not None:
            stop(upstream, upstream_thread)
