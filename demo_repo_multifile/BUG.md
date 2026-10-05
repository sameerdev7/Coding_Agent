# Bug: root cause lives in a different file than the failing traceback

`orders.py`'s `describe_order("shipped")` raises `KeyError: 'shipped'`. The traceback points at
`orders.py`, but the actual bug is a typo in `config.py`'s `STATUS_LABELS` dict — the key is
`"shiped"` (missing a "p") instead of `"shipped"`.

Editing `orders.py` alone fixes nothing. The agent has to notice `orders.py` imports
`STATUS_LABELS` from `config.py`, go look there (`read_file` or `grep`), and fix the typo at
the source.

**Before fix:** `pytest -q` → 1 failed, 2 passed
**After fix:** `pytest -q` → 3 passed

**Difficulty:** hard — forces genuine cross-file navigation instead of single-file pattern matching.
