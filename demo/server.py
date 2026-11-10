"""Optional static server for the graph-to-score page: python demo/server.py."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

STATIC = Path(__file__).resolve().parent / "static"
ROUTES = {
    "/": ("replay.html", "text/html"),
    "/replay": ("replay.html", "text/html"),
    "/replay.js": ("replay.js", "text/javascript"),
    "/replay.css": ("replay.css", "text/css"),
    "/replay-data.js": ("replay-data.js", "text/javascript"),
}


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def serve_asset(self, *, body=True):
        route = ROUTES.get(urlsplit(self.path).path)
        if route is None:
            self.send_error(404)
            return
        filename, mime_type = route
        try:
            content = (STATIC / filename).read_bytes()
        except OSError:
            self.send_error(500, "Static asset unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime_type+"; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(content)

    def do_GET(self):
        self.serve_asset()

    def do_HEAD(self):
        self.serve_asset(body=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    with LocalServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Graph → GNN → Scores: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
