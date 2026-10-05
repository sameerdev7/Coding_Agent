from text_utils import count_vowels, reverse_words


def test_count_vowels_case_insensitive():
    assert count_vowels("Ordinary") == 3


def test_reverse_words_order_not_letters():
    assert reverse_words("hello world") == "world hello"
