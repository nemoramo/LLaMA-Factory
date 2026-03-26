def run_grpo(*args, **kwargs):
    from .workflow import run_grpo as _run_grpo

    return _run_grpo(*args, **kwargs)


__all__ = ["run_grpo"]
