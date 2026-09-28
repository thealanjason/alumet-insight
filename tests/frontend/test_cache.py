import unittest

import pandas as pd

import frontend.cache as cache_module
from frontend.cache import (
    cache_dataframe,
    cache_id_from_store,
    clear_dataframe_cache,
    delete_cached_dataframe,
    df_from_store,
    is_cache_miss,
    load_cached_dataframe,
)


class CacheTests(unittest.TestCase):
    def setUp(self):
        clear_dataframe_cache()
        cache_module._MAX_ENTRIES = cache_module._DEFAULT_MAX_ENTRIES

    def tearDown(self):
        clear_dataframe_cache()
        cache_module._MAX_ENTRIES = cache_module._DEFAULT_MAX_ENTRIES

    def test_cache_dataframe_returns_none_for_empty_input(self):
        self.assertIsNone(cache_dataframe(pd.DataFrame()))
        self.assertIsNone(cache_dataframe(None))

    def test_cache_roundtrip_memory_only_by_default(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=2, freq="s"), "value": [1, 2]}
        )
        cache_id = cache_dataframe(df, prefix="test")

        self.assertIsNotNone(cache_id)
        self.assertTrue(cache_id.startswith("test_"))
        self.assertFalse(cache_module._cache_path(cache_id).exists())

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

    def test_df_from_store_loads_filtered_store_dict(self):
        df = pd.DataFrame({"value": [7]})
        cache_id = cache_dataframe(df, prefix="filtered")
        loaded = df_from_store({"cache_id": cache_id, "metric_order": ["m"]})
        pd.testing.assert_frame_equal(loaded, df)

    def test_load_cached_dataframe_promotes_from_disk_after_memory_clear(self):
        df = pd.DataFrame({"value": [42]})
        cache_id = cache_dataframe(df, prefix="disk", persist_to_disk=True)

        cache_module._MEMORY_CACHE.clear()
        loaded = load_cached_dataframe(cache_id)

        pd.testing.assert_frame_equal(loaded, df)
        self.assertIn(cache_id, cache_module._MEMORY_CACHE)

    def test_memory_only_entry_is_miss_after_memory_clear(self):
        df = pd.DataFrame({"value": [1]})
        cache_id = cache_dataframe(df, prefix="mem")
        cache_module._MEMORY_CACHE.clear()
        self.assertTrue(is_cache_miss(load_cached_dataframe(cache_id)))

    def test_load_cached_dataframe_returns_copy_not_cache_alias(self):
        df = pd.DataFrame({"value": [1, 2]})
        cache_id = cache_dataframe(df, prefix="copy")

        loaded = load_cached_dataframe(cache_id)
        loaded.loc[0, "value"] = 999

        cached = cache_module._MEMORY_CACHE[cache_id]
        self.assertEqual(cached.loc[0, "value"], 1)
        pd.testing.assert_frame_equal(load_cached_dataframe(cache_id), df)

    def test_lru_evicts_oldest_entries(self):
        cache_module._MAX_ENTRIES = 2
        id_a = cache_dataframe(pd.DataFrame({"v": [1]}), prefix="a")
        id_b = cache_dataframe(pd.DataFrame({"v": [2]}), prefix="b")
        id_c = cache_dataframe(pd.DataFrame({"v": [3]}), prefix="c")

        self.assertNotIn(id_a, cache_module._MEMORY_CACHE)
        self.assertIn(id_b, cache_module._MEMORY_CACHE)
        self.assertIn(id_c, cache_module._MEMORY_CACHE)
        self.assertTrue(is_cache_miss(load_cached_dataframe(id_a)))

    def test_lru_refresh_on_load_protects_entry(self):
        cache_module._MAX_ENTRIES = 2
        id_a = cache_dataframe(pd.DataFrame({"v": [1]}), prefix="a")
        id_b = cache_dataframe(pd.DataFrame({"v": [2]}), prefix="b")
        load_cached_dataframe(id_a)  # mark A as recently used
        id_c = cache_dataframe(pd.DataFrame({"v": [3]}), prefix="c")

        self.assertIn(id_a, cache_module._MEMORY_CACHE)
        self.assertNotIn(id_b, cache_module._MEMORY_CACHE)
        self.assertIn(id_c, cache_module._MEMORY_CACHE)

    def test_delete_cached_dataframe_removes_memory_and_disk(self):
        df = pd.DataFrame({"value": [1]})
        cache_id = cache_dataframe(df, prefix="del", persist_to_disk=True)
        path = cache_module._cache_path(cache_id)
        self.assertTrue(path.exists())

        delete_cached_dataframe(cache_id)
        self.assertNotIn(cache_id, cache_module._MEMORY_CACHE)
        self.assertFalse(path.exists())
        self.assertTrue(is_cache_miss(load_cached_dataframe(cache_id)))

    def test_clear_dataframe_cache(self):
        id_a = cache_dataframe(pd.DataFrame({"v": [1]}), prefix="a", persist_to_disk=True)
        id_b = cache_dataframe(pd.DataFrame({"v": [2]}), prefix="b")
        clear_dataframe_cache()
        self.assertEqual(cache_module._MEMORY_CACHE, {})
        self.assertFalse(cache_module._cache_path(id_a).exists())
        self.assertTrue(is_cache_miss(load_cached_dataframe(id_b)))

    def test_cache_id_from_store(self):
        self.assertEqual(cache_id_from_store("abc"), "abc")
        self.assertEqual(cache_id_from_store({"cache_id": "xyz"}), "xyz")
        self.assertIsNone(cache_id_from_store(None))
        self.assertIsNone(cache_id_from_store({"metric_order": []}))


if __name__ == "__main__":
    unittest.main()
