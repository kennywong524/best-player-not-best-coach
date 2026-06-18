"""EDSL execution mode: remote Expected Parrot Jobs vs local API proxy."""

from __future__ import annotations

import os
from typing import Any

# EDSL CONFIG loads repo .env with override=True on first import, clobbering shell exports.
_CENTAUR_TIMEOUT_SEC = "1800"


def ensure_edsl_timeouts() -> None:
    """Keep local-proxy timeouts high; .env often has 300s which causes frontier-model failures."""
    os.environ["EDSL_API_TIMEOUT"] = _CENTAUR_TIMEOUT_SEC
    os.environ["REMOTE_PROXY_TIMEOUT"] = _CENTAUR_TIMEOUT_SEC
    try:
        from edsl.config import CONFIG

        CONFIG.EDSL_API_TIMEOUT = _CENTAUR_TIMEOUT_SEC
    except ImportError:
        pass


def use_remote_inference() -> bool:
    """Default False: local API proxy. Set CENTAUR_EDSL_REMOTE=1 for remote Jobs."""
    return os.environ.get("CENTAUR_EDSL_REMOTE", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def edsl_run_kwargs(
    *,
    description: str,
    visibility: str = "private",
    progress_bar: bool = False,
    verbose: bool = False,
    n: int | None = None,
    cache: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build kwargs for Survey/Jobs.run() using remote Jobs or local API proxy.

  Generation should use cache=False (default) so replicates are independent fresh
  API calls. Judging may pass cache=True to reuse identical pairwise evaluations.
    """
    ensure_edsl_timeouts()
    kwargs: dict[str, Any] = {
        "progress_bar": progress_bar,
        "verbose": verbose,
        "print_exceptions": True,
        "stop_on_exception": False,
        "check_api_keys": False,
        "cache": cache,
        "disable_remote_cache": not cache,
        "fresh": not cache,
        "remote_cache_description": description[:200],
        **extra,
    }
    if n is not None:
        kwargs["n"] = n
    if use_remote_inference():
        kwargs.update(
            {
                "use_api_proxy": False,
                "disable_remote_inference": False,
                "remote_inference_description": description[:200],
                "remote_inference_results_visibility": visibility,
            }
        )
    else:
        kwargs.update(
            {
                "use_api_proxy": True,
                "disable_remote_inference": True,
            }
        )
    return kwargs
