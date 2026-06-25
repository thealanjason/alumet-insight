import unittest

import pandas as pd

import frontend.cache as cache_module
from frontend.cache import (
    cache_dataframe,
    df_from_store,
    ensure_timestamp_datetime,
    is_cache_miss,
    load_cached_dataframe,
)


class CacheTests(unittest.TestCase):
    def test_cache_dataframe_returns_none_for_empty_input(self):
        self.assertIsNone(cache_dataframe(pd.DataFrame()))
        self.assertIsNone(cache_dataframe(None))

    def test_cache_roundtrip_via_memory_and_disk(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="s"), "value": [1, 2]})
        cache_id = cache_dataframe(df, prefix="test")

        self.assertIsNotNone(cache_id)
        self.assertTrue(cache_id.startswith("test_"))

        loaded = load_cached_dataframe(cache_id)
        pd.testing.assert_frame_equal(loaded, df)

    def test_load_cached_dataframe_handles_missing_and_empty_ids(self):
        self.assertTrue(load_cached_dataframe(None).empty)
        loaded = load_cached_dataframe("missing_cache_id")
        self.assertTrue(is_cache_miss(loaded))

    def test_is_cache_miss(self):
        self.assertTrue(is_cache_miss(pd.DataFrame({"__cache_miss__": [True]})))
        self.assertFalse(is_cache_miss(pd.DataFrame({"value": [1]})))
        self.assertFalse(is_cache_miss(pd.DataFrame()))

    def test_df_from_store_handles_cache_id_split_records_and_none(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache_id = cache_dataframe(df, prefix="store")

        from_cache = df_from_store(cache_id)
        pd.testing.assert_frame_equal(from_cache, df)

        split_payload = df.to_dict(orient="split")
        from_split = df_from_store(split_payload)
        pd.testing.assert_frame_equal(from_split, df)

        from_records = df_from_store(df.to_dict(orient="records"))
        pd.testing.assert_frame_equal(from_records, df)

        self.assertTrue(df_from_store(None).empty)

    def test_load_cached_dataframe_promotes_from_disk_after_memory_clear(self):
        df = pd.DataFrame({"value": [42]})
        cache_id = cache_dataframe(df, prefix="disk")

        cache_module._MEMORY_CACHE.clear()
        loaded = load_cached_dataframe(cache_id)

        pd.testing.assert_frame_equal(loaded, df)
        self.assertIn(cache_id, cache_module._MEMORY_CACHE)

    def test_ensure_timestamp_datetime(self):
        raw = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-02"], "value": [1, 2]})
        converted = ensure_timestamp_datetime(raw.copy())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted["timestamp"]))

        already_dt = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "value": [1]})
        unchanged = ensure_timestamp_datetime(already_dt)
        pd.testing.assert_frame_equal(unchanged, already_dt)

    def test_ensure_timestamp_datetime_leaves_df_without_timestamp_unchanged(self):
        df = pd.DataFrame({"value": [1, 2]})
        unchanged = ensure_timestamp_datetime(df.copy())
        pd.testing.assert_frame_equal(unchanged, df)


if __name__ == "__main__":
    unittest.main()
