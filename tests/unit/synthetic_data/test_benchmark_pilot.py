from __future__ import annotations

import hashlib
import json
import unittest

from ai_poet.synthetic_data.benchmark import (
    _result_summary,
    build_fixture_bank,
    select_capacity,
)
from ai_poet.synthetic_data.capacity import (
    PILOT_REPORT_VERSION,
    REPORT_VERSION,
    configured_capacity_plan,
    generation_fingerprint,
    load_capacity_report,
    validate_pilot_gate,
)
from ai_poet.synthetic_data.config import EndpointSettings
from ai_poet.synthetic_data.pilot import (
    BOUNDED_TASK_PILOT_QUOTAS,
    PILOT_QUOTAS,
    select_pilot_poems,
)
from ai_poet.synthetic_data.tasks.base import (
    TASK_MCQ,
    TASK_POEM_COMPLETION,
    TASK_POEM_RECONSTRUCTION,
)
from tests.synthetic_data_helpers import TEST_TMP, make_poem, remove_test_files, settings


def endpoints() -> tuple[EndpointSettings, ...]:
    return tuple(
        EndpointSettings(
            endpoint_id=f"endpoint_{index}",
            endpoint=f"https://endpoint-{index}.test/v1/chat/completions",
            api_key=f"secret-{index}",
            max_concurrency=8,
        )
        for index in range(1, 4)
    )


class BenchmarkAndPilotTests(unittest.TestCase):
    def test_benchmark_result_summary_is_human_readable(self) -> None:
        summary = _result_summary(
            {
                "requests": 100,
                "successes": 98,
                "retryable_errors": 1,
                "nonretryable_errors": 1,
                "requests_per_second": 3.456,
                "p95_latency_seconds": 2.345,
            }
        )

        self.assertEqual(
            summary,
            "98/100 successful, 2 errors, 3.46 req/s, p95 2.35s",
        )

    def test_configured_capacity_plan_uses_endpoint_ceilings(self) -> None:
        generation = settings(endpoints=endpoints())

        plan = configured_capacity_plan(generation)

        self.assertFalse(plan.raw["certified"])
        self.assertEqual(plan.report_fingerprint, "")
        self.assertEqual(
            plan.hard_caps,
            {endpoint.endpoint_id: endpoint.max_concurrency for endpoint in endpoints()},
        )

    def test_fixture_bank_has_the_fixed_eighty_request_mix(self) -> None:
        generation = settings(endpoints=endpoints())
        poems = [
            make_poem(
                verses=tuple(
                    part
                    for couplet in range(1, count + 1)
                    for part in (f"صدر {count}-{couplet}", f"عجز {count}-{couplet}")
                )
            )
            for count in (2, 5, 12, 30, 80)
        ]
        fixtures, probes = build_fixture_bank(poems, generation)
        self.assertEqual(len(fixtures), 80)
        self.assertEqual(
            sum(fixture.request_kind.startswith("instruction") for fixture in fixtures),
            16,
        )
        self.assertEqual(
            sum(fixture.request_kind.startswith("reasoning") for fixture in fixtures),
            64,
        )
        self.assertEqual(len(probes), 4)

    def test_new_task_fixture_banks_are_task_shaped(self) -> None:
        generation = settings(endpoints=endpoints())
        poems = [make_poem()]
        mcq_fixtures, mcq_probes = build_fixture_bank(
            poems, generation, TASK_MCQ
        )
        reconstruction_fixtures, reconstruction_probes = build_fixture_bank(
            poems, generation, TASK_POEM_RECONSTRUCTION
        )
        completion_fixtures, completion_probes = build_fixture_bank(
            poems, generation, TASK_POEM_COMPLETION
        )
        self.assertEqual(len(mcq_fixtures), 80)
        self.assertEqual(len(reconstruction_fixtures), 80)
        self.assertEqual(len(completion_fixtures), 80)
        self.assertFalse(mcq_probes)
        self.assertFalse(reconstruction_probes)
        self.assertFalse(completion_probes)
        self.assertEqual(
            sum(fixture.request_kind == "mcq_validation" for fixture in mcq_fixtures),
            40,
        )
        completion_reasoning = next(
            fixture
            for fixture in completion_fixtures
            if fixture.request_kind == "reasoning_generation"
        )
        self.assertIn("بداية القصيدة:", completion_reasoning.messages[-1]["content"])
        self.assertEqual(
            sum(
                fixture.request_kind == "reconstruction_validation"
                for fixture in reconstruction_fixtures
            ),
            40,
        )

    def test_generation_fingerprints_are_task_isolated(self) -> None:
        generation = settings(endpoints=endpoints())
        self.assertNotEqual(
            generation_fingerprint(generation, TASK_MCQ),
            generation_fingerprint(generation, TASK_POEM_RECONSTRUCTION),
        )
        self.assertNotEqual(
            generation_fingerprint(generation, TASK_POEM_COMPLETION),
            generation_fingerprint(generation, TASK_POEM_RECONSTRUCTION),
        )

    def test_capacity_selection_uses_smallest_level_at_ninety_five_percent(self) -> None:
        selected, nonconverged = select_capacity(
            [
                {"safe": True, "concurrency": 1, "requests_per_second": 10.0},
                {"safe": True, "concurrency": 2, "requests_per_second": 19.1},
                {"safe": True, "concurrency": 4, "requests_per_second": 20.0},
            ]
        )
        self.assertEqual(selected, 2)
        self.assertFalse(nonconverged)

    def test_pilot_selection_obeys_all_quotas_and_includes_largest(self) -> None:
        generation = settings(max_source_chars=10_000, endpoints=endpoints())
        poems = []
        index = 0
        sizes = {
            "couplets_1_3": 2,
            "couplets_4_9": 5,
            "couplets_10_24": 12,
            "couplets_25_74": 30,
            "couplets_75_plus": 80,
        }
        for name, quota in PILOT_QUOTAS.items():
            if name == "oversized":
                continue
            for _ in range(quota):
                index += 1
                count = sizes[name]
                poems.append(
                    make_poem(
                        verses=tuple(
                            part
                            for couplet in range(count)
                            for part in (
                                f"صدر {index}-{couplet}",
                                f"عجز {index}-{couplet}",
                            )
                        )
                    )
                )
        oversized = []
        for length in (10_001, 10_100, 10_200, 10_300, 12_000):
            index += 1
            poem = make_poem(verses=("س" * length, f"عجز {index}"))
            poems.append(poem)
            oversized.append(poem)

        selected, groups = select_pilot_poems(poems, generation)
        self.assertEqual(len(selected), 300)
        self.assertEqual(
            {name: len(sample_ids) for name, sample_ids in groups.items()},
            PILOT_QUOTAS,
        )
        largest = max(oversized, key=lambda poem: len(poem.poem_text))
        self.assertIn(largest.sample_id, groups["oversized"])

    def test_new_task_pilot_is_bounded_to_twenty_four_couplets(self) -> None:
        generation = settings(endpoints=endpoints())
        poems = []
        serial = 0
        for count, quota in ((2, 120), (5, 100), (12, 80)):
            for _ in range(quota):
                serial += 1
                poems.append(
                    make_poem(
                        verses=tuple(
                            part
                            for couplet in range(count)
                            for part in (
                                f"صدر {serial}-{couplet}",
                                f"عجز {serial}-{couplet}",
                            )
                        )
                    )
                )
        poems.append(
            make_poem(
                verses=tuple(
                    part
                    for couplet in range(30)
                    for part in (f"صدر طويل {couplet}", f"عجز طويل {couplet}")
                )
            )
        )
        selected, groups = select_pilot_poems(poems, generation, TASK_MCQ)
        self.assertEqual(len(selected), 300)
        self.assertEqual(
            {name: len(sample_ids) for name, sample_ids in groups.items()},
            BOUNDED_TASK_PILOT_QUOTAS,
        )
        self.assertTrue(all(poem.couplet_count <= 24 for poem in selected))

    def test_completion_pilot_excludes_single_couplet_poems(self) -> None:
        generation = settings(endpoints=endpoints())
        poems = [make_poem(verses=("صدر وحيد", "عجز وحيد"))]
        serial = 0
        for count, quota in ((2, 120), (5, 100), (12, 80)):
            for _ in range(quota):
                serial += 1
                poems.append(
                    make_poem(
                        verses=tuple(
                            part
                            for couplet in range(count)
                            for part in (
                                f"صدر {serial}-{couplet}",
                                f"عجز {serial}-{couplet}",
                            )
                        )
                    )
                )

        selected, groups = select_pilot_poems(
            poems, generation, TASK_POEM_COMPLETION
        )
        self.assertEqual(len(selected), 300)
        self.assertTrue(all(poem.couplet_count >= 2 for poem in selected))
        self.assertEqual(
            {name: len(sample_ids) for name, sample_ids in groups.items()},
            BOUNDED_TASK_PILOT_QUOTAS,
        )

    def test_capacity_and_pilot_artifacts_are_fingerprint_checked(self) -> None:
        generation = settings(endpoints=endpoints())
        capacity_name = "capacity_test.json"
        pilot_name = "pilot_gate_test.json"
        review_name = "pilot_review_test.json"
        remove_test_files(capacity_name, pilot_name, review_name)
        self.addCleanup(
            remove_test_files, capacity_name, pilot_name, review_name
        )
        TEST_TMP.mkdir(exist_ok=True)
        capacity_payload = {
            "report_version": REPORT_VERSION,
            "certified": True,
            "generation_fingerprint": generation_fingerprint(generation),
            "source_sha256": "source",
            "endpoints": [
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "endpoint": endpoint.endpoint,
                    "model": endpoint.model or generation.model,
                    "selected_concurrency": 4,
                    "latency_baselines": {"reasoning_generation": 2.0},
                }
                for endpoint in endpoints()
            ],
        }
        capacity_path = TEST_TMP / capacity_name
        capacity_path.write_text(json.dumps(capacity_payload), encoding="utf-8")
        plan = load_capacity_report(
            capacity_path, generation, source_sha256="source"
        )
        self.assertEqual(plan.total_capacity, 12)
        self.assertNotIn("secret-1", capacity_path.read_text("utf-8"))

        pilot_payload = {
            "report_version": PILOT_REPORT_VERSION,
            "passed": True,
            "source_sha256": "source",
            "generation_fingerprint": generation_fingerprint(generation),
            "capacity_report_fingerprint": plan.report_fingerprint,
            "content_fingerprint": "pilot-content",
        }
        pilot_bytes = json.dumps(pilot_payload).encode("utf-8")
        pilot_path = TEST_TMP / pilot_name
        pilot_path.write_bytes(pilot_bytes)
        review_path = TEST_TMP / review_name
        review_path.write_text(
            json.dumps(
                {
                    "pilot_report_fingerprint": hashlib.sha256(
                        pilot_bytes
                    ).hexdigest(),
                    "pilot_content_fingerprint": "pilot-content",
                    "reviews": [
                        {"sample_id": f"sample-{index}", "approved": True}
                        for index in range(30)
                    ],
                }
            ),
            encoding="utf-8",
        )
        fingerprints = validate_pilot_gate(
            pilot_path,
            review_path,
            settings=generation,
            source_sha256="source",
            capacity_fingerprint=plan.report_fingerprint,
        )
        self.assertEqual(len(fingerprints), 2)
