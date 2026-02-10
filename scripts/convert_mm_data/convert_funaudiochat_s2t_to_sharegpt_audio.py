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

r"""Deprecated wrapper.

Use `convert_funaudiochat_s2t_to_qwen3_asr_sharegpt_audio.py` instead.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType


_NEW = Path(__file__).with_name("convert_funaudiochat_s2t_to_qwen3_asr_sharegpt_audio.py")


def _load_new() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_lf_qwen3_asr_converter", _NEW)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {_NEW}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    warnings.warn(
        "`convert_funaudiochat_s2t_to_sharegpt_audio.py` is deprecated; "
        "use `convert_funaudiochat_s2t_to_qwen3_asr_sharegpt_audio.py` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _load_new().main()


if __name__ == "__main__":
    main()
