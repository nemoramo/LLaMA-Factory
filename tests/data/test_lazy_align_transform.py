# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle

import pytest
from datasets import Dataset

from llamafactory.data.loader import _LazyAlignTransform


class _DummyConverter:
    def __call__(self, example):
        # Mimic sharegpt/openai-like list fields.
        return {"_prompt": example["messages"], "_response": [{"role": "assistant", "content": example["text"]}]}


@pytest.mark.runs_on(["cpu"])
def test_lazy_align_transform_single_example():
    transform = _LazyAlignTransform(dataset_converter=_DummyConverter(), id_key="id")
    out = transform({"id": 1, "messages": [{"role": "user", "content": "hi"}], "text": "ok"})
    assert out["id"] == 1
    assert out["_prompt"] == [{"role": "user", "content": "hi"}]
    assert out["_response"] == [{"role": "assistant", "content": "ok"}]


@pytest.mark.runs_on(["cpu"])
def test_lazy_align_transform_batched_examples():
    transform = _LazyAlignTransform(dataset_converter=_DummyConverter(), id_key="id")
    out = transform(
        {
            "id": [1, 2],
            "messages": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            "text": ["x", "y"],
        }
    )
    assert out["id"] == [1, 2]
    assert out["_prompt"] == [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    assert out["_response"] == [[{"role": "assistant", "content": "x"}], [{"role": "assistant", "content": "y"}]]


@pytest.mark.runs_on(["cpu"])
def test_lazy_align_with_transform_unbatching():
    raw = Dataset.from_dict(
        {
            "id": [1, 2],
            "messages": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            "text": ["x", "y"],
        }
    )
    transform = _LazyAlignTransform(dataset_converter=_DummyConverter(), id_key="id")
    ds = raw.with_transform(transform, output_all_columns=False)

    ex0 = ds[0]
    assert isinstance(ex0["id"], int)
    assert ex0["id"] == 1
    assert ex0["_prompt"] == [{"role": "user", "content": "a"}]
    assert ex0["_response"] == [{"role": "assistant", "content": "x"}]

    exb = ds[:2]
    assert exb["id"] == [1, 2]
    assert exb["_prompt"] == [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]


@pytest.mark.runs_on(["cpu"])
def test_lazy_align_transform_picklable():
    transform = _LazyAlignTransform(dataset_converter=_DummyConverter(), id_key="id")
    pickle.dumps(transform)


@pytest.mark.runs_on(["cpu"])
def test_lazy_align_transform_messages_only_single_example():
    class _MsgOnlyConverter:
        def __call__(self, example):
            return {"_prompt": example["messages"], "_response": []}

    transform = _LazyAlignTransform(dataset_converter=_MsgOnlyConverter(), id_key=None)
    out = transform({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]})

    assert out["_prompt"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert out["_response"] == []
