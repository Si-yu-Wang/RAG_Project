"""Check the optional static server without model or project-data dependencies."""
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Handler, LocalServer, STATIC


class StaticServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = LocalServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_only_current_page_and_its_assets_are_served(self):
        for route, filename in (("/", "replay.html"), ("/replay", "replay.html"), ("/replay.js", "replay.js"),
                                ("/replay.css", "replay.css"), ("/replay-data.js", "replay-data.js")):
            with self.subTest(route=route), urlopen(self.base+route) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), (STATIC / filename).read_bytes())
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_retired_pages_apis_and_arbitrary_files_are_not_served(self):
        for route in ("/live", "/flow", "/project", "/api/project", "/api/metrics", "/api/flow/status",
                      "/source?path=README.md", "/server.py", "/../README.md", "/%2e%2e/server.py"):
            with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                urlopen(self.base+route)
            self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base+"/api/flow/runs", data=b"{}"))
        self.assertEqual(error.exception.code, 501)

    def test_head_serves_headers_without_the_body(self):
        with urlopen(Request(self.base+"/", method="HEAD")) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(int(response.headers["Content-Length"]), (STATIC / "replay.html").stat().st_size)
            self.assertEqual(response.read(), b"")


if __name__ == "__main__":
    unittest.main()
