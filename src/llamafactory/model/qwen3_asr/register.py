from __future__ import annotations


def register_qwen3_asr() -> None:
    """Register Qwen3-ASR classes with transformers AutoClasses.

    Qwen3-ASR is not yet in upstream `transformers` (as of the repo's pinned version).
    We vendor the minimal config/model/processor implementations under `llamafactory.model.qwen3_asr`.
    """

    from transformers import AutoConfig, AutoModel, AutoProcessor

    try:
        from transformers import AutoModelForSeq2SeqLM  # type: ignore
    except Exception:  # noqa: BLE001
        AutoModelForSeq2SeqLM = None  # type: ignore[assignment]

    from .configuration_qwen3_asr import Qwen3ASRConfig
    from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from .processing_qwen3_asr import Qwen3ASRProcessor

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass

    try:
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except ValueError:
        pass

    if AutoModelForSeq2SeqLM is not None:
        try:
            AutoModelForSeq2SeqLM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)  # type: ignore[attr-defined]
        except ValueError:
            pass

    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except ValueError:
        pass

