from __future__ import annotations

import unittest
from unittest.mock import patch

from ai_poet.synthetic_data.cli import build_parser
from ai_poet.synthetic_data.config import load_generation_settings, load_run_settings

class ConfigurationTests(unittest.TestCase):
    def test_settings_load_required_values_from_dotenv(self) -> None:
        args = build_parser().parse_args([])
        environment = {
            "GEMMA_ENDPOINT": "https://env.example/v1/chat/completions",
            "GEMMA_MODEL": "env-model",
            "GEMMA_API_KEY": "env-secret",
        }

        with (
            patch("ai_poet.synthetic_data.config.load_dotenv") as load_dotenv,
            patch.dict("os.environ", environment, clear=True),
        ):
            result = load_generation_settings(args)

        load_dotenv.assert_called_once_with()
        self.assertEqual(result.endpoint, environment["GEMMA_ENDPOINT"])
        self.assertEqual(result.model, environment["GEMMA_MODEL"])
        self.assertEqual(result.api_key, environment["GEMMA_API_KEY"])

    def test_settings_require_every_dotenv_value(self) -> None:
        args = build_parser().parse_args([])
        environment = {
            "GEMMA_ENDPOINT": "https://env.example/v1/chat/completions",
            "GEMMA_MODEL": "env-model",
            "GEMMA_API_KEY": "env-secret",
        }

        for missing_name in environment:
            with self.subTest(missing_name=missing_name):
                incomplete = environment | {missing_name: ""}
                with (
                    patch("ai_poet.synthetic_data.config.load_dotenv"),
                    patch.dict("os.environ", incomplete, clear=True),
                    self.assertRaisesRegex(ValueError, missing_name),
                ):
                    load_generation_settings(args)

    def test_parser_has_no_endpoint_or_model_override(self) -> None:
        help_text = build_parser().format_help()
        self.assertNotIn("--endpoint", help_text)
        self.assertNotIn("--model", help_text)

    def test_run_settings_reject_invalid_concurrency_and_limit(self) -> None:
        invalid_arguments = (("--concurrency", "0"), ("--limit", "0"))
        for option, value in invalid_arguments:
            with self.subTest(option=option):
                args = build_parser().parse_args([option, value])
                with self.assertRaises(ValueError):
                    load_run_settings(args)
