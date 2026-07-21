import unittest

from ai_trading.runtime_settings import ConfigurationError, get_runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_are_local_paper_and_no_synthetic_data(self):
        settings = get_runtime_settings({})
        self.assertEqual(settings.trading_mode, "paper")
        self.assertFalse(settings.live_trading_enabled)
        self.assertFalse(settings.allow_synthetic_data)
        self.assertEqual(settings.api_bind_host, "127.0.0.1")
        self.assertFalse(settings.debug)
        self.assertFalse(settings.live_orders_enabled)

    def test_live_orders_need_mode_flag_and_both_credentials(self):
        environment = {
            "TRADING_MODE": "live",
            "LIVE_TRADING_ENABLED": "true",
            "BINANCE_API_KEY": "key",
        }
        self.assertFalse(get_runtime_settings(environment).live_orders_enabled)
        environment["BINANCE_API_SECRET"] = "secret"
        self.assertTrue(get_runtime_settings(environment).live_orders_enabled)

    def test_invalid_boolean_and_port_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            get_runtime_settings({"APP_DEBUG": "maybe"})
        with self.assertRaises(ConfigurationError):
            get_runtime_settings({"API_PORT": "70000"})


if __name__ == "__main__":
    unittest.main()
