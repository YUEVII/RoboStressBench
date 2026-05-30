from __future__ import annotations

import os
import warnings


def configure_runtime_noise() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*CUDA initialization: The NVIDIA driver on your system is too old.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*tied weights mapping and config for this model specifies to tie.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Setting `pad_token_id` to `eos_token_id`.*",
    )

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass
