# Bug: two logic errors, not missing validation

`text_utils.py` has two functions that are subtly *wrong*, not just missing edge-case guards:

1. `count_vowels` only checks lowercase vowels, so an uppercase vowel (the "O" in "Ordinary") isn't counted. `count_vowels("Ordinary")` returns `2`, expected `3`.
2. `reverse_words` reverses the entire string character-by-character (`sentence[::-1]`) instead of reversing word order. `reverse_words("hello world")` returns `"dlrow olleh"`, expected `"world hello"`.

**Before fix:** `pytest -q` → 2 failed
**After fix:** `pytest -q` → 2 passed

**Difficulty:** medium — requires actually reading and reasoning about the logic, not pattern-matching "add a guard clause."
