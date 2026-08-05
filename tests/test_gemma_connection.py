from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import URLError

from generate_sft import (
    GemmaClient,
    GemmaConnectionError,
    GenerationSettings,
    PoemRecord,
    build_parser,
    run,
)


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
            patch("generate_sft.urlopen", side_effect=URLError("offline")) as request,
            patch("generate_sft.random.random", return_value=0.0),
            patch("generate_sft.time.sleep") as sleep,
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
            meter_id=1,
            meter_name="test meter",
            verses=("first half", "second half"),
            metadata_conflict=False,
        )
        args = build_parser().parse_args(
            ["--input", "unused.parquet", "--output-dir", "unused-output"]
        )

        self.assertEqual(args.max_network_retries, 3)
        with (
            patch("generate_sft._settings", return_value=settings()),
            patch("generate_sft.load_poems", return_value=[poem]),
            patch("generate_sft.file_sha256", return_value="source-digest"),
            patch(
                "generate_sft.generate_one",
                side_effect=GemmaConnectionError("Gemma is offline"),
            ),
            patch.object(Path, "mkdir"),
            patch("generate_sft.write_jsonl"),
            patch("generate_sft.write_outputs") as write_outputs,
            redirect_stdout(io.StringIO()),
            self.assertRaises(GemmaConnectionError),
        ):
            run(args)

        write_outputs.assert_not_called()

