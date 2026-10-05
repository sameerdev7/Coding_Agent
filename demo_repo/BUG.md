# Bug: missing input validation

`calculator.py`'s `divide` and `average` don't guard against invalid input:

- `divide(a, 0)` raises `ZeroDivisionError` instead of `ValueError`
- `average([])` raises `ZeroDivisionError` instead of `ValueError`

**Before fix:** `pytest -q` → 2 failed
**After fix:** `pytest -q` → 2 passed

**Difficulty:** easy — single file, two independent guard clauses. Good warm-up demo.
