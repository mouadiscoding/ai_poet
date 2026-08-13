from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from ai_poet.synthetic_data.cli import build_parser, main
from ai_poet.synthetic_data.config import (
    DEFAULT_MAX_COUPLETS,
    load_generation_settings,
    load_run_settings,
)

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

    def test_settings_load_exactly_three_indexed_endpoints(self) -> None:
        args = build_parser().parse_args([])
        environment = {"GEMMA_MODEL": "gemma-test"}
        for index in range(1, 4):
            environment[f"GEMMA_ENDPOINT_{index}"] = (
                f"https://endpoint-{index}.test/v1/chat/completions"
            )
            environment[f"GEMMA_API_KEY_{index}"] = f"secret-{index}"
            environment[f"GEMMA_MAX_CONCURRENCY_{index}"] = str(index * 2)

        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
        ):
            result = load_generation_settings(args)

        self.assertTrue(result.is_multi_endpoint)
        self.assertEqual(
            [endpoint.max_concurrency for endpoint in result.configured_endpoints],
            [2, 4, 6],
        )
        self.assertEqual(result.secrets, ("secret-1", "secret-2", "secret-3"))

    def test_settings_load_per_endpoint_model_aliases(self) -> None:
        args = build_parser().parse_args([])
        environment = {}
        for index in range(1, 4):
            environment[f"GEMMA_ENDPOINT_{index}"] = f"https://endpoint-{index}.test"
            environment[f"GEMMA_MODEL_{index}"] = f"gemma-alias-{index}"
            environment[f"GEMMA_API_KEY_{index}"] = f"secret-{index}"

        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
        ):
            result = load_generation_settings(args)

        self.assertEqual(result.model, "gemma-alias-1")
        self.assertEqual(
            [endpoint.model for endpoint in result.configured_endpoints],
            ["gemma-alias-1", "gemma-alias-2", "gemma-alias-3"],
        )

    def test_settings_reject_partial_or_mixed_indexed_models(self) -> None:
        args = build_parser().parse_args([])
        environment = {}
        for index in range(1, 4):
            environment[f"GEMMA_ENDPOINT_{index}"] = f"https://endpoint-{index}.test"
            environment[f"GEMMA_API_KEY_{index}"] = f"secret-{index}"
        environment["GEMMA_MODEL_1"] = "gemma-alias-1"

        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
            self.assertRaisesRegex(ValueError, "GEMMA_MODEL_2"),
        ):
            load_generation_settings(args)

        environment["GEMMA_MODEL"] = "shared-model"
        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
            self.assertRaisesRegex(ValueError, "Do not mix GEMMA_MODEL"),
        ):
            load_generation_settings(args)

    def test_settings_reject_partial_or_mixed_indexed_configuration(self) -> None:
        args = build_parser().parse_args([])
        partial = {
            "GEMMA_MODEL": "gemma-test",
            "GEMMA_ENDPOINT_1": "https://endpoint-1.test",
            "GEMMA_API_KEY_1": "secret-1",
        }
        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", partial, clear=True),
            self.assertRaisesRegex(ValueError, "GEMMA_ENDPOINT_2"),
        ):
            load_generation_settings(args)

        mixed = partial | {
            "GEMMA_ENDPOINT": "https://legacy.test",
            "GEMMA_API_KEY": "legacy-secret",
        }
        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", mixed, clear=True),
            self.assertRaisesRegex(ValueError, "Do not mix"),
        ):
            load_generation_settings(args)

    def test_parser_has_no_endpoint_or_model_override(self) -> None:
        help_text = build_parser().format_help()
        self.assertNotIn("--endpoint", help_text)
        self.assertNotIn("--model", help_text)

    def test_run_settings_reject_invalid_numeric_bounds(self) -> None:
        invalid_arguments = (
            ("--concurrency", "0"),
            ("--limit", "0"),
            ("--max-couplets", "0"),
        )
        for option, value in invalid_arguments:
            with self.subTest(option=option):
                args = build_parser().parse_args([option, value])
                with self.assertRaises(ValueError):
                    load_run_settings(args)

    def test_max_couplets_has_bounded_default_and_override(self) -> None:
        self.assertEqual(
            build_parser().parse_args([]).max_couplets,
            DEFAULT_MAX_COUPLETS,
        )
        self.assertEqual(
            build_parser().parse_args(["--max-couplets", "12"]).max_couplets,
            12,
        )

    def test_skip_pilot_review_allows_missing_gate_artifacts(self) -> None:
        args = build_parser().parse_args(["--skip-pilot-review"])
        environment = {"GEMMA_MODEL": "gemma-test"}
        for index in range(1, 4):
            environment[f"GEMMA_ENDPOINT_{index}"] = f"https://endpoint-{index}.test"
            environment[f"GEMMA_API_KEY_{index}"] = f"secret-{index}"

        with (
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
        ):
            result = load_run_settings(args)

        self.assertFalse(result.enforce_pilot_gate)
        self.assertIsNone(result.capacity_report)
        self.assertIsNone(result.pilot_report)
        self.assertIsNone(result.pilot_review)

    def test_skip_pilot_review_emits_yellow_warning(self) -> None:
        environment = {"GEMMA_MODEL": "gemma-test"}
        for index in range(1, 4):
            environment[f"GEMMA_ENDPOINT_{index}"] = f"https://endpoint-{index}.test"
            environment[f"GEMMA_API_KEY_{index}"] = f"secret-{index}"
        stderr = io.StringIO()

        with (
            patch("sys.argv", ["ai-poet-generate-sft", "--skip-pilot-review"]),
            patch("ai_poet.synthetic_data.config.load_dotenv"),
            patch.dict("os.environ", environment, clear=True),
            patch("ai_poet.synthetic_data.cli.run", return_value=0),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 0)
        self.assertIn("\033[33mWARNING:", stderr.getvalue())
        self.assertIn("configured endpoint concurrency limits", stderr.getvalue())
        self.assertIn("\033[0m", stderr.getvalue())
