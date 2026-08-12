from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest

from ai_poet.synthetic_data.checkpoint import CheckpointWriter, load_checkpoint_state
from ai_poet.synthetic_data.generation import generate_one
from tests.synthetic_data_helpers import (
    TEST_TMP,
    QueueClient,
    make_poem,
    remove_test_files,
    settings,
    valid_instruction_value,
    valid_pipeline_outputs,
    valid_reasoning_value,
    valid_verdict,
)


class PartialResumeTests(unittest.TestCase):
    def test_accepted_instruction_and_chunk_resume_without_api_calls(self) -> None:
        poem = make_poem()
        path = TEST_TMP / "partial_checkpoint.jsonl"
        TEST_TMP.mkdir(exist_ok=True)
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        first = QueueClient(valid_pipeline_outputs(poem))
        record = generate_one(
            poem,
            first,
            settings(),
            generation_fingerprint="contract",
            checkpoint_writer=CheckpointWriter(path),
        )
        state = load_checkpoint_state(path)
        self.assertIn(poem.sample_id, state.instructions)
        self.assertIn(0, state.reasoning_chunks[poem.sample_id])

        resumed_client = QueueClient([])
        resumed = generate_one(
            poem,
            resumed_client,
            settings(),
            generation_fingerprint="contract",
            resume_instruction=state.instructions[poem.sample_id],
            resume_chunks=state.reasoning_chunks[poem.sample_id],
        )
        self.assertFalse(resumed_client.calls)
        self.assertEqual(resumed["instruction"], record["instruction"])
        self.assertEqual(resumed["response"], record["response"])

    def test_parallel_chunks_are_bounded_and_assembled_in_source_order(self) -> None:
        verses = tuple(
            part
            for index in range(1, 9)
            for part in (f"صدر البيت {index}", f"عجز البيت {index}")
        )
        poem = make_poem(verses=verses)

        class ContextClient:
            tracer = None
            total_effective_capacity = 8

            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.maximum = 0

            def chat(self, _messages, **kwargs):
                kind = kwargs["trace_context"]["request_kind"]
                if kind == "instruction_generation":
                    return json.dumps(valid_instruction_value(poem), ensure_ascii=False)
                if kind in {"instruction_validation", "reasoning_validation"}:
                    return json.dumps(valid_verdict(), ensure_ascii=False)
                if kind == "reasoning_generation":
                    start = kwargs["trace_context"]["chunk_start"] - 1
                    end = kwargs["trace_context"]["chunk_end"]
                    with self.lock:
                        self.active += 1
                        self.maximum = max(self.maximum, self.active)
                    time.sleep(0.02 * (9 - end))
                    with self.lock:
                        self.active -= 1
                    return json.dumps(
                        valid_reasoning_value(
                            poem,
                            start_offset=start,
                            chunk_size=end - start,
                        ),
                        ensure_ascii=False,
                    )
                raise AssertionError(f"unexpected request kind {kind}")

        client = ContextClient()
        with ThreadPoolExecutor(max_workers=4) as executor:
            record = generate_one(
                poem,
                client,
                settings(),
                chunk_executor=executor,
                chunk_parallelism=2,
            )

        self.assertEqual(client.maximum, 2)
        self.assertEqual(record["reasoning_chunk_count"], 3)
        positions = [record["response"].index(f"البيت {index}:") for index in range(1, 9)]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(record["response"].endswith(poem.poem_text))
