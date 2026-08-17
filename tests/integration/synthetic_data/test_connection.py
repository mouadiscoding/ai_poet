"""Integration tests for transport retry and runner abort behavior."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import URLError

from ai_poet.synthetic_data.cli import build_parser
from ai_poet.synthetic_data.client import GemmaClient
from ai_poet.synthetic_data.config import GenerationSettings, RunSettings
from ai_poet.synthetic_data.errors import GemmaConnectionError
from ai_poet.synthetic_data.poems import PoemRecord
from ai_poet.synthetic_data.runner import run


def settings() -> GenerationSettings:
    return GenerationSettings(
        endpoint="https://example.test/v1/chat/completions",
        model="gemma-test",
        api_key="secret",
        insecure=False,
        timeout=1,
        max_network_retries=3,
        max_repairs=2,
        temperature=0.4,
        top_p=0.9,
        max_tokens=4096,
        min_chars=80,
        max_source_chars=24000,
        chunk_chars=12000,
    )


class GemmaConnectionTests(unittest.TestCase):
    def test_connection_failure_retries_three_times_then_raises(self) -> None:
        client = GemmaClient(settings())

        with (
            patch(
                "ai_poet.synthetic_data.client.urlopen",
                side_effect=URLError("offline"),
            ) as request,
            patch("ai_poet.synthetic_data.client.random.random", return_value=0.0),
            patch("ai_poet.synthetic_data.client.time.sleep") as sleep,
            self.assertRaisesRegex(
                GemmaConnectionError,
                "connection failed after 3 retries",
            ),
        ):
            client.chat([{"role": "user", "content": "test"}])

        self.assertEqual(request.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])

    def test_run_aborts_on_exhausted_connection_failure(self) -> None:
        poem = PoemRecord(
            sample_id="1" * 64,
            source_row_indices=(1,),
            source_urls=("https://example.test/poem",),
            poet_name="test poet",
            poem_title="test poem",
            poem_theme="test theme",
            meter_id=1,
            meter_name="test meter",
            verses=("first half", "second half"),
            metadata_conflict=False,
        )
        args = build_parser().parse_args(
            ["--input", "unused.parquet", "--output-dir", "unused-output"]
        )
        run_settings = RunSettings(
            input=args.input,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            limit=args.limit,
            trace=args.trace,
            generation=settings(),
        )

        self.assertEqual(args.max_network_retries, 3)
        with (
            patch("ai_poet.synthetic_data.runner.load_poems", return_value=[poem]),
            patch("ai_poet.synthetic_data.runner.file_sha256", return_value="source-digest"),
            patch(
                "ai_poet.synthetic_data.runner.generate_one",
                side_effect=GemmaConnectionError("Gemma is offline"),
            ),
            patch.object(Path, "mkdir"),
            patch("ai_poet.synthetic_data.runner.write_jsonl"),
            patch("ai_poet.synthetic_data.runner.write_outputs") as write_outputs,
            redirect_stdout(io.StringIO()),
            self.assertRaises(GemmaConnectionError),
        ):
            run(run_settings)

        write_outputs.assert_not_called()

