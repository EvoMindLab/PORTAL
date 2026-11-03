"""Per-parameter random-stream derivation for the Generator tab.

Each randomized quantity in a component (centre ``c``, ``kappa_+``/``kappa_-``,
``p_+``/``p_-``, rotation, and each transform vector parameter) draws from its
**own** independent stream rather than one shared RNG. A stream is determined by
``(seed, param_key, counter)``, all stored on the component so generation is
fully reproducible:

- ``seed``    — the component's user-facing seed.
- ``param_key`` — a stable string identifying the stream (e.g. ``"kappa_plus"``
                or ``"tf0:mu_plus"`` for the first transform's mu_+).
- ``counter`` — the parameter's reroll counter; bumping it by one yields a fresh,
                independent draw for that one stream only ("reroll this row").

The streams are independent and reproducible because they are mixed through
``numpy.random.SeedSequence``, which is designed for exactly this. The result is
a legacy ``RandomState`` (not a ``Generator``) so the engine's existing samplers
— which call ``rand``/``beta``/``randint``/``permutation`` — keep working.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _stream_id(param_key: str) -> int:
    """Stable, process-independent integer id for a stream key.

    ``hash()`` is salted per process, so a content hash is used instead to keep
    generation reproducible across runs and machines.
    """
    digest = hashlib.blake2b(param_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def derive_param_rng(seed: int, param_key: str,
                     counter: int = 0) -> np.random.RandomState:
    """Return the independent ``RandomState`` for one parameter stream."""
    ss = np.random.SeedSequence(
        [int(seed), _stream_id(str(param_key)), int(counter)]
    )
    return np.random.RandomState(ss.generate_state(1))
