import asyncio
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_trading.api import application, run


class ApiFacadeTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop(application._LEGACY_MODULE_NAME, None)
        logging.shutdown()

    def test_package_import_exposes_run_without_loading_heavy_legacy_module(self):
        self.assertTrue(callable(run))
        self.assertNotIn(application._LEGACY_MODULE_NAME, sys.modules)

    def test_facade_loads_and_delegates_to_a_legacy_module(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "legacy_api.py"
            module_path.write_text(
                "called = False\n"
                "app = object()\n"
                "def run():\n"
                "    global called\n"
                "    called = True\n",
                encoding="utf-8",
            )
            with patch.object(application, "LEGACY_API_PATH", module_path):
                application.run()
                module = sys.modules[application._LEGACY_MODULE_NAME]
                self.assertTrue(module.called)
                self.assertIs(application.get_app(), module.app)

    def test_compose_uses_the_package_runner_not_runpy(self):
        compose = Path("docker/docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('command: ["python3", "-m", "ai_trading.api.run"]', compose)
        self.assertNotIn("runpy.run_path", compose)

    def test_facade_loads_the_actual_health_endpoint_without_ml_dependencies(self):
        async def call_health():
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await application.get_app()(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": "/health",
                    "raw_path": b"/health",
                    "query_string": b"",
                    "headers": [],
                    "client": ("127.0.0.1", 12345),
                    "server": ("testserver", 80),
                },
                receive,
                send,
            )
            return messages

        messages = asyncio.run(call_health())
        response = next(message for message in messages if message["type"] == "http.response.start")
        body = next(message for message in messages if message["type"] == "http.response.body")
        payload = json.loads(body["body"])
        self.assertEqual(response["status"], 200)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("timestamp", payload)

    def test_package_exposes_app_lazily_for_uvicorn_import_strings(self):
        import ai_trading.api as api_package

        self.assertIs(api_package.app, application.get_app())


if __name__ == "__main__":
    unittest.main()
