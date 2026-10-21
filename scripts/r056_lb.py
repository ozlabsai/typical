#!/usr/bin/env python3
"""R056 eval: least-busy reverse proxy over N identical decision-server replicas (stdlib).

The native server serialises forwards per process, so WorkflowEvals + JevBench run against several replicas
(2 per GPU) behind one URL. GET is answered by the first backend; POST goes to the backend with the fewest
requests in flight.   python3 scripts/r056_lb.py 8090 http://127.0.0.1:8100 http://127.0.0.1:8101 ...
"""
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = sys.argv[2:]
BUSY = {b: 0 for b in BACKENDS}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def relay(self, backend, body=None):
        headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization")}
        req = urllib.request.Request(backend + self.path, body, headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                code, data, ctype = resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as err:
            code, data, ctype = err.code, err.read(), err.headers.get("Content-Type", "application/json")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.relay(BACKENDS[0])

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with LOCK:
            backend = min(BACKENDS, key=BUSY.__getitem__)
            BUSY[backend] += 1
        try:
            self.relay(backend, body)
        finally:
            with LOCK:
                BUSY[backend] -= 1


ThreadingHTTPServer.request_queue_size = 256  # default backlog 5 resets bursts (R055 note)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
