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

"""FunAudioChat model registration.

Author: yufeng.ma
"""

_REGISTERED = False


def register_funaudiochat() -> None:
    """Register FunAudioChat classes with transformers AutoClasses.

    This is idempotent and safe to call multiple times.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

    from .configuration_funaudiochat import FunAudioChatAudioEncoderConfig, FunAudioChatConfig
    from .modeling_funaudiochat import FunAudioChatForConditionalGeneration
    from .processing_funaudiochat import FunAudioChatProcessor

    for model_type, cfg_cls in [
        ("funaudiochat", FunAudioChatConfig),
        ("funaudiochat_audio_encoder", FunAudioChatAudioEncoderConfig),
    ]:
        try:
            AutoConfig.register(model_type, cfg_cls)
        except ValueError:
            pass

    try:
        AutoProcessor.register(FunAudioChatConfig, FunAudioChatProcessor)
    except ValueError:
        pass

    try:
        AutoModelForSeq2SeqLM.register(FunAudioChatConfig, FunAudioChatForConditionalGeneration)
    except ValueError:
        pass

    _REGISTERED = True
