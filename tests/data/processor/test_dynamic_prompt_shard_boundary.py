def test_dynamic_prompt_prefetch_flushes_carryover_at_shard_boundary_when_buffer_produces_no_packs():
    # Regression test for a subtle data-loss bug:
    # - With carryover_packs enabled, the packer may intentionally hold back some segments (carry_items)
    #   for cross-buffer mixing.
    # - If a shard-boundary is hit in a later buffer AND that buffer produces no packed outputs
    #   (e.g., all examples are dropped), carry_items must still be flushed before clearing;
    #   otherwise they are silently discarded.
    from llamafactory.data.processor.dynamic_prompt import (
        SHARD_BOUNDARY_KEY,
        _DynamicPromptPackedPrefetchDataset,
    )

    class _FakePacker:
        def encode_examples(self, examples):
            ids = list(examples.get("id") or [])
            # Simulate "all examples dropped" for this buffer (e.g., encode error).
            if any(str(x).startswith("bad") for x in ids):
                return [], []
            items = [{"x": str(x)} for x in ids]
            lengths = [1 for _ in items]
            return items, lengths

        def pack_encoded_items(self, items, lengths, *, carryover_packs=0):
            if not items:
                return {}, [], []
            # Minimal packing semantics for the dataset wrapper:
            # - When carryover_packs>0, hold back the second half as carry_items.
            # - When carryover_packs==0, emit everything.
            if int(carryover_packs or 0) > 0 and len(items) > 1:
                mid = len(items) // 2
                out_items = items[:mid]
                carry_items = items[mid:]
            else:
                out_items = items
                carry_items = []
            packed = {"x": [it["x"] for it in out_items]} if out_items else {}
            carry_lengths = [1 for _ in carry_items]
            return packed, carry_items, carry_lengths

    def _raw_rows():
        # Buffer #1: produces carryover (c,d) that should NOT be dropped.
        for i in ["a", "b", "c", "d"]:
            yield {"id": i}

        # Buffer #2: hits shard boundary and all rows are "dropped" by encode_examples.
        yield {"id": "bad1"}
        yield {"id": "bad2"}
        yield {SHARD_BOUNDARY_KEY: True}

    ds = _DynamicPromptPackedPrefetchDataset(
        iterable_ds=[],  # unused for this unit test
        packer=_FakePacker(),
        buffer_size=4,
        prefetch_buffers=2,
        carryover_packs=1,
    )

    outs = list(ds._iter_one_pass_with_prefetch(iter(_raw_rows())))
    xs = [ex["x"] for ex in outs]
    assert xs == ["a", "b", "c", "d"]
