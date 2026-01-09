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

"""FunAudioChat model package.

Author: yufeng.ma
"""

from .configuration_funaudiochat import FunAudioChatAudioEncoderConfig, FunAudioChatConfig
from .modeling_funaudiochat import FunAudioChatForConditionalGeneration
from .processing_funaudiochat import FunAudioChatProcessor
from .register import register_funaudiochat


__all__ = [
    "FunAudioChatAudioEncoderConfig",
    "FunAudioChatConfig",
    "FunAudioChatForConditionalGeneration",
    "FunAudioChatProcessor",
    "register_funaudiochat",
]
