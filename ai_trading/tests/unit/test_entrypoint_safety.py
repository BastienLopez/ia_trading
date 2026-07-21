import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class EntrypointSafetyTests(unittest.TestCase):
    def test_runtime_entrypoints_use_shared_settings(self):
        for relative_path in ("ai_trading/api.py", "web_app/app.py"):
            source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
            self.assertIn("get_runtime_settings", source)
            ast.parse(source)

    def test_example_contains_no_demo_secret_or_real_credential_placeholder(self):
        source = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("dev_key_very_secret", source)
        self.assertNotIn("votre_vraie", source)
        self.assertIn("TRADING_MODE=paper", source)
        self.assertIn("LIVE_TRADING_ENABLED=false", source)


if __name__ == "__main__":
    unittest.main()
