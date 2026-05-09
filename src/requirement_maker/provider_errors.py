from dataclasses import dataclass
from enum import Enum


class ProviderErrorKind(str, Enum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    PROVIDER = "provider"
    RETRY_EXHAUSTED = "retry_exhausted"


class ProviderStage(str, Enum):
    TRANSCRIPTION = "transcription"
    REQUIREMENT_GENERATION = "requirement generation"


_SECRET_MARKERS = (
    "sk-",
    "sk-ant-",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "Authorization",
    "Bearer ",
)


@dataclass(frozen=True)
class ProviderError(RuntimeError):
    stage: ProviderStage
    kind: ProviderErrorKind
    provider: str
    detail: str
    transient: bool

    def __str__(self) -> str:
        return self.detail


def classify_provider_exception(
    exc: BaseException,
    *,
    stage: ProviderStage,
    provider: str,
) -> ProviderError:
    name = exc.__class__.__name__.lower()
    message = _safe_detail(str(exc))
    status_code = getattr(exc, "status_code", None)

    if _has_any(name, "authentication", "permissiondenied", "unauthorized") or status_code in {
        401,
        403,
    }:
        return ProviderError(
            stage=stage,
            kind=ProviderErrorKind.AUTHENTICATION,
            provider=provider,
            detail="provider rejected credentials",
            transient=False,
        )
    if "ratelimit" in name or status_code == 429:
        return ProviderError(
            stage=stage,
            kind=ProviderErrorKind.RATE_LIMIT,
            provider=provider,
            detail="provider rate limit was reached",
            transient=True,
        )
    if _has_any(name, "timeout", "apitimeout"):
        return ProviderError(
            stage=stage,
            kind=ProviderErrorKind.TIMEOUT,
            provider=provider,
            detail="provider request timed out",
            transient=True,
        )
    if _has_any(name, "badrequest", "invalidrequest", "unprocessable") or status_code in {
        400,
        422,
    }:
        return ProviderError(
            stage=stage,
            kind=ProviderErrorKind.INVALID_REQUEST,
            provider=provider,
            detail=message or "provider rejected the request",
            transient=False,
        )
    if status_code is not None and int(status_code) >= 500:
        return ProviderError(
            stage=stage,
            kind=ProviderErrorKind.PROVIDER,
            provider=provider,
            detail="provider service error",
            transient=True,
        )
    return ProviderError(
        stage=stage,
        kind=ProviderErrorKind.PROVIDER,
        provider=provider,
        detail=message or "provider request failed",
        transient=False,
    )


def retry_exhausted(error: ProviderError) -> ProviderError:
    return ProviderError(
        stage=error.stage,
        kind=ProviderErrorKind.RETRY_EXHAUSTED,
        provider=error.provider,
        detail=f"retry limit exhausted after transient {error.kind.value.replace('_', ' ')}",
        transient=False,
    )


def format_provider_cli_error(error: ProviderError) -> str:
    guidance = {
        ProviderErrorKind.AUTHENTICATION: (
            "Check the relevant API key in your environment or .env file."
        ),
        ProviderErrorKind.RATE_LIMIT: (
            "Wait and retry, lower --concurrency, or increase provider quota."
        ),
        ProviderErrorKind.TIMEOUT: "Retry with a larger --timeout or a smaller input.",
        ProviderErrorKind.INVALID_REQUEST: (
            "Check the selected model/options and provider request limits."
        ),
        ProviderErrorKind.PROVIDER: "Retry later or check provider service status.",
        ProviderErrorKind.RETRY_EXHAUSTED: (
            "Retry later, reduce --concurrency, increase --timeout, or raise --retries."
        ),
    }[error.kind]
    permanence = "transient" if error.transient else "permanent"
    return (
        f"{error.stage.value.capitalize()} failed ({error.provider} {error.kind.value}, "
        f"{permanence}): {error.detail}\nRepair: {guidance}"
    )


def _has_any(value: str, *needles: str) -> bool:
    return any(needle in value for needle in needles)


def _safe_detail(detail: str) -> str:
    if not detail:
        return ""
    if any(marker in detail for marker in _SECRET_MARKERS):
        return "provider returned an error; details were hidden because they may contain secrets"
    return " ".join(detail.strip().split())[:240]
