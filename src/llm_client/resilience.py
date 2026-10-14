"""Backoff and the circuit breaker. No I/O here, so both are easy to test."""

import random
import time

from .config import BreakerSettings, RetrySettings
from .errors import RateLimitError, RetryableError


def next_delay(err: RetryableError, attempt: int, cfg: RetrySettings) -> float | None:
    """
    Seconds to wait before retrying the same provider, or None to give up on
    it and move to the next one. `attempt` is 0 for the first try.
    """
    if attempt + 1 >= cfg.max_attempts:
        return None

    if isinstance(err, RateLimitError):
        if err.quota_exhausted:
            return None                       # daily quota: seconds won't fix it
        if err.retry_after is not None:
            if err.retry_after > cfg.max_retry_after_s:
                return None                   # too long to make the user wait
            # Believe the provider, plus a little jitter: every client that got
            # the same header would otherwise retry in the same instant.
            return err.retry_after + random.uniform(0, cfg.base_delay_s)

    # Full jitter: a random point between 0 and the exponential ceiling.
    # Spreads out clients that all failed at the same moment (thundering herd).
    ceiling = min(cfg.max_delay_s, cfg.base_delay_s * 2 ** attempt)
    return random.uniform(0, ceiling)


class CircuitBreaker:
    """
    closed     normal, calls go through
    open       too many consecutive failures, calls skip this provider
    half-open  cooldown passed, one test call goes through:
               success → closed, failure → open again
    """

    def __init__(self, cfg: BreakerSettings, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self.trial_in_flight = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.cfg.cooldown_s:
            return "half-open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half-open" and not self.trial_in_flight:
            self.trial_in_flight = True       # only one test call at a time
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.trial_in_flight = False

    def record_failure(self) -> None:
        self.failures += 1
        self.trial_in_flight = False
        if self.opened_at is not None or self.failures >= self.cfg.failure_threshold:
            self.opened_at = self.clock()     # (re)open, restart the cooldown
