from __future__ import annotations

import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from ai_poet.synthetic_data.assignment import choose_template, sft_split
from ai_poet.synthetic_data.corpus import load_poems
from ai_poet.synthetic_data.meters import meter_name
from ai_poet.synthetic_data.poems import format_poem, poem_hash, split_poem_chunks
from tests.synthetic_data_helpers import TEST_TMP, make_poem, remove_test_files

class CorpusTests(unittest.TestCase):
    def test_meter_mapping_and_poem_format(self) -> None:
        self.assertEqual(meter_name(0), "البسيط")
        self.assertEqual(meter_name(14), "النثر")
        with self.assertRaises(ValueError):
            meter_name(17)
        with self.assertRaises(ValueError):
            meter_name(-1)
        self.assertEqual(format_poem(("صدر", "عجز")), "صدر = عجز")
        with self.assertRaises(ValueError):
            format_poem(("شطر وحيد",))

    def test_hash_split_is_stable_and_template_choice_is_random(self) -> None:
        poem = make_poem()
        self.assertEqual(poem_hash(poem.verses), poem.sample_id)
        self.assertEqual(sft_split(poem.sample_id), sft_split(poem.sample_id))
        with patch(
            "ai_poet.synthetic_data.assignment.random.choice",
            side_effect=lambda templates: templates[4],
        ) as choice:
            self.assertEqual(choose_template().template_id, "occasion_addressee")
        choice.assert_called_once()

    def test_chunks_preserve_couplet_boundaries(self) -> None:
        verses = ("أ" * 10, "ب" * 10, "ج" * 10, "د" * 10)
        chunks = split_poem_chunks(verses, 25)
        self.assertEqual(chunks, ["أ" * 10 + " = " + "ب" * 10, "ج" * 10 + " = " + "د" * 10])

    def test_load_poems_deduplicates_and_uses_meter_majority(self) -> None:
        verses = ["صدر البيت", "عجز البيت"]
        rows = [
            {
                "poem_title": None,
                "poem_meter": 0,
                "poem_verses": verses,
                "poem_url": "https://example.test/a",
                "poet_name": "شاعر",
            },
            {
                "poem_title": "عنوان",
                "poem_meter": 10,
                "poem_verses": verses,
                "poem_url": "https://example.test/b",
                "poet_name": "شاعر",
            },
            {
                "poem_title": "عنوان",
                "poem_meter": 10,
                "poem_verses": verses,
                "poem_url": "https://example.test/c",
                "poet_name": "شاعر",
            },
        ]
        path = TEST_TMP / "source.parquet"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        try:
            pq.write_table(pa.Table.from_pylist(rows), path)
            poems = load_poems(path)
        finally:
            remove_test_files(path.name)
        self.assertEqual(len(poems), 1)
        self.assertEqual(poems[0].meter_name, "المديد")
        self.assertTrue(poems[0].metadata_conflict)
        self.assertEqual(len(poems[0].source_urls), 3)
